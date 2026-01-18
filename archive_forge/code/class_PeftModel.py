from __future__ import annotations
import collections
import inspect
import os
import warnings
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Optional, Union
import packaging.version
import torch
import transformers
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules
from accelerate.utils import get_balanced_memory
from huggingface_hub import ModelCard, ModelCardData, hf_hub_download
from safetensors.torch import save_file as safe_save_file
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput, SequenceClassifierOutput, TokenClassifierOutput
from transformers.utils import PushToHubMixin
from . import __version__
from .config import PeftConfig
from .tuners import (
from .utils import (
class PeftModel(PushToHubMixin, torch.nn.Module):
    """
    Base model encompassing various Peft methods.

    Args:
        model ([`~transformers.PreTrainedModel`]): The base transformer model used for Peft.
        peft_config ([`PeftConfig`]): The configuration of the Peft model.
        adapter_name (`str`,  *optional*): The name of the adapter, defaults to `"default"`.

    **Attributes**:
        - **base_model** ([`torch.nn.Module`]) -- The base transformer model used for Peft.
        - **peft_config** ([`PeftConfig`]) -- The configuration of the Peft model.
        - **modules_to_save** (`list` of `str`) -- The list of sub-module names to save when
            saving the model.
        - **prompt_encoder** ([`PromptEncoder`]) -- The prompt encoder used for Peft if
            using [`PromptLearningConfig`].
        - **prompt_tokens** (`torch.Tensor`) -- The virtual prompt tokens used for Peft if
            using [`PromptLearningConfig`].
        - **transformer_backbone_name** (`str`) -- The name of the transformer
            backbone in the base model if using [`PromptLearningConfig`].
        - **word_embeddings** (`torch.nn.Embedding`) -- The word embeddings of the transformer backbone
            in the base model if using [`PromptLearningConfig`].
    """

    def __init__(self, model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str='default') -> None:
        super().__init__()
        self.modules_to_save = None
        self.active_adapter = adapter_name
        self.peft_type = peft_config.peft_type
        self._is_prompt_learning = peft_config.is_prompt_learning
        if self._is_prompt_learning:
            self._peft_config = {adapter_name: peft_config}
            self.base_model = model
            self.add_adapter(adapter_name, peft_config)
        else:
            self._peft_config = None
            cls = PEFT_TYPE_TO_MODEL_MAPPING[peft_config.peft_type]
            self.base_model = cls(model, {adapter_name: peft_config}, adapter_name)
            self.set_additional_trainable_modules(peft_config, adapter_name)
        if getattr(model, 'is_gradient_checkpointing', True):
            model = self._prepare_model_for_gradient_checkpointing(model)
        if hasattr(self.base_model, 'config') and hasattr(self.base_model.config, 'pretraining_tp'):
            self.base_model.config.pretraining_tp = 1

    @property
    def peft_config(self) -> dict[str, PeftConfig]:
        if self._is_prompt_learning:
            return self._peft_config
        return self.base_model.peft_config

    @property
    def active_adapters(self) -> list[str]:
        try:
            adapters = self.base_model.active_adapters
        except AttributeError:
            adapters = self.active_adapter
            if isinstance(adapters, str):
                adapters = [adapters]
        return adapters

    @peft_config.setter
    def peft_config(self, value: dict[str, PeftConfig]):
        if self._is_prompt_learning:
            self._peft_config = value
        else:
            self.base_model.peft_config = value

    def save_pretrained(self, save_directory: str, safe_serialization: bool=True, selected_adapters: Optional[list[str]]=None, save_embedding_layers: Union[str, bool]='auto', is_main_process: bool=True, **kwargs: Any) -> None:
        """
        This function saves the adapter model and the adapter configuration files to a directory, so that it can be
        reloaded using the [`PeftModel.from_pretrained`] class method, and also used by the [`PeftModel.push_to_hub`]
        method.

        Args:
            save_directory (`str`):
                Directory where the adapter model and configuration files will be saved (will be created if it does not
                exist).
            safe_serialization (`bool`, *optional*):
                Whether to save the adapter files in safetensors format, defaults to `True`.
            selected_adapters (`List[str]`,  *optional*):
                A list of adapters to be saved. If `None`, will default to all adapters.
            save_embedding_layers (`Union[bool, str]`, *optional*, defaults to `"auto"`):
                If `True`, save the embedding layers in addition to adapter weights. If `auto`, checks the common
                embedding layers `peft.utils.other.EMBEDDING_LAYER_NAMES` in config's `target_modules` when available.
                and automatically sets the boolean flag. This only works for 🤗 transformers models.
            is_main_process (`bool`, *optional*):
                Whether the process calling this is the main process or not. Will default to `True`. Will not save the
                checkpoint if not on the main process, which is important for multi device setups (e.g. DDP).
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the `push_to_hub` method.
        """
        if os.path.isfile(save_directory):
            raise ValueError(f'Provided path ({save_directory}) should be a directory, not a file')
        if selected_adapters is None:
            selected_adapters = list(self.peft_config.keys())
        elif any((selected_adapter_name not in list(self.peft_config.keys()) for selected_adapter_name in selected_adapters)):
            raise ValueError(f'You passed an invalid `selected_adapters` arguments, current supported adapter names are {list(self.peft_config.keys())} - got {selected_adapters}.')
        if is_main_process:
            os.makedirs(save_directory, exist_ok=True)
            self.create_or_update_model_card(save_directory)
        for adapter_name in selected_adapters:
            peft_config = self.peft_config[adapter_name]
            output_state_dict = get_peft_model_state_dict(self, state_dict=kwargs.get('state_dict', None), adapter_name=adapter_name, save_embedding_layers=save_embedding_layers)
            output_dir = os.path.join(save_directory, adapter_name) if adapter_name != 'default' else save_directory
            os.makedirs(output_dir, exist_ok=True)
            if is_main_process and safe_serialization:
                ptrs = collections.defaultdict(list)
                for name, tensor in output_state_dict.items():
                    if isinstance(tensor, torch.Tensor):
                        ptrs[id_tensor_storage(tensor)].append(name)
                    else:
                        ptrs[id(tensor)].append(name)
                shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}
                for _, names in shared_ptrs.items():
                    for shared_tensor_name in names[1:]:
                        output_state_dict[shared_tensor_name] = output_state_dict[shared_tensor_name].clone()
                safe_save_file(output_state_dict, os.path.join(output_dir, SAFETENSORS_WEIGHTS_NAME), metadata={'format': 'pt'})
            elif is_main_process:
                torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))
            if peft_config.base_model_name_or_path is None:
                peft_config.base_model_name_or_path = self.base_model.__dict__.get('name_or_path', None) if peft_config.is_prompt_learning else self.base_model.model.__dict__.get('name_or_path', None)
            inference_mode = peft_config.inference_mode
            peft_config.inference_mode = True
            if peft_config.task_type is None:
                base_model_class = self._get_base_model_class(is_prompt_tuning=peft_config.is_prompt_learning)
                parent_library = base_model_class.__module__
                auto_mapping_dict = {'base_model_class': base_model_class.__name__, 'parent_library': parent_library}
            else:
                auto_mapping_dict = None
            if is_main_process:
                peft_config.save_pretrained(output_dir, auto_mapping_dict=auto_mapping_dict)
            peft_config.inference_mode = inference_mode

    @classmethod
    def from_pretrained(cls, model: torch.nn.Module, model_id: Union[str, os.PathLike], adapter_name: str='default', is_trainable: bool=False, config: Optional[PeftConfig]=None, **kwargs: Any) -> PeftModel:
        """
        Instantiate a PEFT model from a pretrained model and loaded PEFT weights.

        Note that the passed `model` may be modified inplace.

        Args:
            model ([`torch.nn.Module`]):
                The model to be adapted. For 🤗 Transformers models, the model should be initialized with the
                [`~transformers.PreTrainedModel.from_pretrained`].
            model_id (`str` or `os.PathLike`):
                The name of the PEFT configuration to use. Can be either:
                    - A string, the `model id` of a PEFT configuration hosted inside a model repo on the Hugging Face
                      Hub.
                    - A path to a directory containing a PEFT configuration file saved using the `save_pretrained`
                      method (`./my_peft_config_directory/`).
            adapter_name (`str`, *optional*, defaults to `"default"`):
                The name of the adapter to be loaded. This is useful for loading multiple adapters.
            is_trainable (`bool`, *optional*, defaults to `False`):
                Whether the adapter should be trainable or not. If `False`, the adapter will be frozen and can only be
                used for inference.
            config ([`~peft.PeftConfig`], *optional*):
                The configuration object to use instead of an automatically loaded configuration. This configuration
                object is mutually exclusive with `model_id` and `kwargs`. This is useful when configuration is already
                loaded before calling `from_pretrained`.
            kwargs: (`optional`):
                Additional keyword arguments passed along to the specific PEFT configuration class.
        """
        from .mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING, PEFT_TYPE_TO_CONFIG_MAPPING
        if config is None:
            config = PEFT_TYPE_TO_CONFIG_MAPPING[PeftConfig._get_peft_type(model_id, subfolder=kwargs.get('subfolder', None), revision=kwargs.get('revision', None), cache_dir=kwargs.get('cache_dir', None), use_auth_token=kwargs.get('use_auth_token', None), token=kwargs.get('token', None))].from_pretrained(model_id, **kwargs)
        elif isinstance(config, PeftConfig):
            config.inference_mode = not is_trainable
        else:
            raise ValueError(f'The input config must be a PeftConfig, got {config.__class__}')
        if getattr(model, 'hf_device_map', None) is not None and len(set(model.hf_device_map.values()).intersection({'cpu', 'disk'})) > 0:
            remove_hook_from_submodules(model)
        if config.is_prompt_learning and is_trainable:
            raise ValueError('Cannot set a prompt learning adapter to trainable when loading pretrained adapter.')
        else:
            config.inference_mode = not is_trainable
        if config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys():
            model = cls(model, config, adapter_name)
        else:
            model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[config.task_type](model, config, adapter_name)
        model.load_adapter(model_id, adapter_name, is_trainable=is_trainable, **kwargs)
        return model

    def _setup_prompt_encoder(self, adapter_name: str):
        config = self.peft_config[adapter_name]
        if not hasattr(self, 'prompt_encoder'):
            self.prompt_encoder = torch.nn.ModuleDict({})
            self.prompt_tokens = {}
        transformer_backbone = None
        for name, module in self.base_model.named_children():
            for param in module.parameters():
                param.requires_grad = False
            if isinstance(module, PreTrainedModel):
                if transformer_backbone is None:
                    transformer_backbone = module
                    self.transformer_backbone_name = name
        if transformer_backbone is None:
            transformer_backbone = self.base_model
        if config.num_transformer_submodules is None:
            config.num_transformer_submodules = 2 if config.task_type == TaskType.SEQ_2_SEQ_LM else 1
        for named_param, value in list(transformer_backbone.named_parameters()):
            deepspeed_distributed_tensor_shape = getattr(value, 'ds_shape', None)
            if value.shape[0] == self.base_model.config.vocab_size or (deepspeed_distributed_tensor_shape is not None and deepspeed_distributed_tensor_shape[0] == self.base_model.config.vocab_size):
                self.word_embeddings = transformer_backbone.get_submodule(named_param.replace('.weight', ''))
                break
        if config.peft_type == PeftType.PROMPT_TUNING:
            prompt_encoder = PromptEmbedding(config, self.word_embeddings)
        elif config.peft_type == PeftType.MULTITASK_PROMPT_TUNING:
            prompt_encoder = MultitaskPromptEmbedding(config, self.word_embeddings)
        elif config.peft_type == PeftType.P_TUNING:
            prompt_encoder = PromptEncoder(config)
        elif config.peft_type == PeftType.PREFIX_TUNING:
            prompt_encoder = PrefixEncoder(config)
        else:
            raise ValueError('Not supported')
        prompt_encoder = prompt_encoder.to(self.device)
        self.prompt_encoder.update(torch.nn.ModuleDict({adapter_name: prompt_encoder}))
        self.prompt_tokens[adapter_name] = torch.arange(config.num_virtual_tokens * config.num_transformer_submodules).long()

    def _prepare_model_for_gradient_checkpointing(self, model: PreTrainedModel):
        """
        Prepares the model for gradient checkpointing if necessary
        """
        if not (getattr(model, 'is_loaded_in_8bit', False) or getattr(model, 'is_loaded_in_4bit', False) or getattr(model, 'is_quantized', False)):
            if hasattr(model, 'enable_input_require_grads'):
                model.enable_input_require_grads()
            elif hasattr(model, 'get_input_embeddings'):

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        return model

    def get_prompt_embedding_to_save(self, adapter_name: str) -> torch.Tensor:
        """
        Returns the prompt embedding to save when saving the model. Only applicable when using a prompt learning
        method.
        """
        prompt_encoder = self.prompt_encoder[adapter_name]
        prompt_tokens = self.prompt_tokens[adapter_name].unsqueeze(0).expand(1, -1).to(prompt_encoder.embedding.weight.device)
        if self.peft_config[adapter_name].peft_type == PeftType.PREFIX_TUNING:
            prompt_tokens = prompt_tokens[:, :self.peft_config[adapter_name].num_virtual_tokens]
        if self.peft_config[adapter_name].peft_type == PeftType.MULTITASK_PROMPT_TUNING:
            prompt_embeddings = super(MultitaskPromptEmbedding, prompt_encoder).forward(prompt_tokens)
        else:
            prompt_embeddings = prompt_encoder(prompt_tokens)
        return prompt_embeddings[0].detach().cpu()

    def get_prompt(self, batch_size: int, task_ids: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Returns the virtual prompts to use for Peft. Only applicable when using a prompt learning method.
        """
        peft_config = self.active_peft_config
        prompt_encoder = self.prompt_encoder[self.active_adapter]
        prompt_tokens = self.prompt_tokens[self.active_adapter].unsqueeze(0).expand(batch_size, -1).to(prompt_encoder.embedding.weight.device)
        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            prompt_tokens = prompt_tokens[:, :peft_config.num_virtual_tokens]
            if peft_config.inference_mode:
                past_key_values = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
            else:
                past_key_values = prompt_encoder(prompt_tokens)
            if self.base_model_torch_dtype is not None:
                past_key_values = past_key_values.to(self.base_model_torch_dtype)
            past_key_values = past_key_values.view(batch_size, peft_config.num_virtual_tokens, peft_config.num_layers * 2, peft_config.num_attention_heads, peft_config.token_dim // peft_config.num_attention_heads)
            if peft_config.num_transformer_submodules == 2:
                past_key_values = torch.cat([past_key_values, past_key_values], dim=2)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(peft_config.num_transformer_submodules * 2)
            if TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING.get(self.config.model_type, None) is not None:
                post_process_fn = TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING[self.config.model_type]
                past_key_values = post_process_fn(past_key_values)
            return past_key_values
        else:
            if peft_config.peft_type == PeftType.MULTITASK_PROMPT_TUNING:
                prompts = prompt_encoder(prompt_tokens, task_ids)
            elif peft_config.inference_mode:
                prompts = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
            else:
                prompts = prompt_encoder(prompt_tokens)
            return prompts

    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        """
        Returns the number of trainable parameters and the number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            if num_params == 0 and hasattr(param, 'ds_numel'):
                num_params = param.ds_numel
            if param.__class__.__name__ == 'Params4bit':
                num_params = num_params * 2
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        return (trainable_params, all_param)

    def print_trainable_parameters(self) -> None:
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params, all_param = self.get_nb_trainable_parameters()
        print(f'trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}')

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def forward(self, *args: Any, **kwargs: Any):
        """
        Forward pass of the model.
        """
        return self.get_base_model()(*args, **kwargs)

    def _get_base_model_class(self, is_prompt_tuning=False):
        """
        Returns the base model class.
        """
        if not is_prompt_tuning:
            return self.base_model.model.__class__
        return self.base_model.__class__

    @contextmanager
    def disable_adapter(self):
        """
        Context manager that disables the adapter module. Use this to run inference on the base model.

        Example:

        ```py
        >>> with model.disable_adapter():
        ...     model(inputs)
        ```
        """
        try:
            if self.peft_config[self.active_adapter].is_prompt_learning:
                old_forward = self.forward
                self.forward = self.base_model.forward
                old_prepare_inputs_for_generation = self.prepare_inputs_for_generation
                self.prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
            else:
                self.base_model.disable_adapter_layers()
            yield
        finally:
            if self.peft_config[self.active_adapter].is_prompt_learning:
                self.forward = old_forward
                self.prepare_inputs_for_generation = old_prepare_inputs_for_generation
            else:
                self.base_model.enable_adapter_layers()

    def get_base_model(self) -> torch.nn.Module:
        """
        Returns the base model.
        """
        return self.base_model if self.active_peft_config.is_prompt_learning or self.peft_type == PeftType.POLY else self.base_model.model

    def add_adapter(self, adapter_name: str, peft_config: PeftConfig) -> None:
        """
        Add an adapter to the model based on the passed configuration.

        The name for the new adapter should be unique.

        The new adapter is not automatically set as the active adapter. Use [`PeftModel.set_adapter`] to set the active
        adapter.

        Args:
            adapter_name (`str`):
                The name of the adapter to be added.
            peft_config ([`PeftConfig`]):
                The configuration of the adapter to be added.
        """
        if peft_config.peft_type != self.peft_type:
            raise ValueError(f'Cannot combine adapters with different peft types. Found {self.peft_type} and {peft_config.peft_type}.')
        try:
            if peft_config.is_prompt_learning:
                self.peft_config[adapter_name] = peft_config
                if hasattr(self.config, 'to_dict'):
                    dict_config = self.config.to_dict()
                else:
                    dict_config = self.config
                peft_config = _prepare_prompt_learning_config(peft_config, dict_config)
                self._setup_prompt_encoder(adapter_name)
            elif peft_config.is_adaption_prompt:
                self.base_model.add_adapter(adapter_name, peft_config)
            else:
                self.peft_config[adapter_name] = peft_config
                self.base_model.inject_adapter(self.base_model.model, adapter_name)
        except Exception:
            if adapter_name in self.peft_config:
                del self.peft_config[adapter_name]
            raise
        self.set_additional_trainable_modules(peft_config, adapter_name)

    def set_additional_trainable_modules(self, peft_config, adapter_name):
        if getattr(peft_config, 'modules_to_save', None) is not None:
            if self.modules_to_save is None:
                self.modules_to_save = set(peft_config.modules_to_save)
            else:
                self.modules_to_save.update(peft_config.modules_to_save)
            _set_trainable(self, adapter_name)

    @classmethod
    def _split_kwargs(cls, kwargs: dict[str, Any]):
        _kwargs_not_in_hf_hub_download_signature = ('use_auth_token',)
        hf_hub_download_kwargs = {}
        other_kwargs = {}
        for key, value in kwargs.items():
            if key in inspect.signature(hf_hub_download).parameters or key in _kwargs_not_in_hf_hub_download_signature:
                hf_hub_download_kwargs[key] = value
            else:
                other_kwargs[key] = value
        return (hf_hub_download_kwargs, other_kwargs)

    def load_adapter(self, model_id: str, adapter_name: str, is_trainable: bool=False, **kwargs: Any):
        """
        Load a trained adapter into the model.

        The name for the new adapter should be unique.

        The new adapter is not automatically set as the active adapter. Use [`PeftModel.set_adapter`] to set the active
        adapter.

        Args:
            adapter_name (`str`):
                The name of the adapter to be added.
            peft_config ([`PeftConfig`]):
                The configuration of the adapter to be added.
            is_trainable (`bool`, *optional*, defaults to `False`):
                Whether the adapter should be trainable or not. If `False`, the adapter will be frozen and can only be
                used for inference.
            kwargs: (`optional`):
                Additional arguments to modify the way the adapter is loaded, e.g. the token for Hugging Face Hub.
        """
        from .mapping import PEFT_TYPE_TO_CONFIG_MAPPING
        hf_hub_download_kwargs, kwargs = self._split_kwargs(kwargs)
        torch_device = infer_device()
        if adapter_name not in self.peft_config:
            peft_config = PEFT_TYPE_TO_CONFIG_MAPPING[PeftConfig._get_peft_type(model_id, **hf_hub_download_kwargs)].from_pretrained(model_id, **hf_hub_download_kwargs)
            if peft_config.is_prompt_learning and is_trainable:
                raise ValueError('Cannot set a prompt learning adapter to trainable when loading pretrained adapter.')
            else:
                peft_config.inference_mode = not is_trainable
            self.add_adapter(adapter_name, peft_config)
        adapters_weights = load_peft_weights(model_id, device=torch_device, **hf_hub_download_kwargs)
        load_result = set_peft_model_state_dict(self, adapters_weights, adapter_name=adapter_name)
        if getattr(self, 'hf_device_map', None) is not None and len(set(self.hf_device_map.values()).intersection({'cpu', 'disk'})) > 0 and (len(self.peft_config) == 1):
            device_map = kwargs.get('device_map', 'auto')
            max_memory = kwargs.get('max_memory', None)
            offload_dir = kwargs.get('offload_folder', None)
            offload_index = kwargs.get('offload_index', None)
            dispatch_model_kwargs = {}
            if 'offload_index' in inspect.signature(dispatch_model).parameters:
                dispatch_model_kwargs['offload_index'] = offload_index
            no_split_module_classes = self._no_split_modules
            if device_map != 'sequential':
                max_memory = get_balanced_memory(self, max_memory=max_memory, no_split_module_classes=no_split_module_classes, low_zero=device_map == 'balanced_low_0')
            if isinstance(device_map, str):
                device_map = infer_auto_device_map(self, max_memory=max_memory, no_split_module_classes=no_split_module_classes)
            dispatch_model(self, device_map=device_map, offload_dir=offload_dir, **dispatch_model_kwargs)
            hook = AlignDevicesHook(io_same_device=True)
            if self.peft_config[adapter_name].is_prompt_learning:
                remove_hook_from_submodules(self.prompt_encoder)
            add_hook_to_module(self.get_base_model(), hook)
        if not is_trainable:
            self.eval()
        return load_result

    def set_adapter(self, adapter_name: str) -> None:
        """
        Sets the active adapter.

        Only one adapter can be active at a time.

        Additionally, this function will set the specified adapter to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str`):
                The name of the adapter to be set as active. The adapter must be loaded first.
        """
        if adapter_name not in self.peft_config:
            raise ValueError(f'Adapter {adapter_name} not found.')
        self.active_adapter = adapter_name
        if not self.peft_config[adapter_name].is_prompt_learning:
            self.base_model.set_adapter(adapter_name)
        _set_adapter(self, adapter_name)

    @property
    def base_model_torch_dtype(self):
        return getattr(self.base_model, 'dtype', None)

    @property
    def active_peft_config(self):
        return self.peft_config[self.active_adapter]

    def create_or_update_model_card(self, output_dir: str):
        """
        Updates or create model card to include information about peft:
        1. Adds `peft` library tag
        2. Adds peft version
        3. Adds base model info
        4. Adds quantization information if it was used
        """
        filename = os.path.join(output_dir, 'README.md')
        card = ModelCard.load(filename) if os.path.exists(filename) else ModelCard.from_template(ModelCardData())
        card.data['library_name'] = 'peft'
        model_config = getattr(self, 'config', None)
        if hasattr(model_config, 'to_dict'):
            model_config = model_config.to_dict()
        if model_config is not None and '_name_or_path' in model_config:
            card.data['base_model'] = model_config['_name_or_path']
        lines = card.text.splitlines()
        quantization_config = None
        if hasattr(model_config, 'quantization_config'):
            quantization_config = self.config.quantization_config.to_dict()
        training_config_text = ''
        quantization_prefix = 'The following `bitsandbytes` quantization config was used during training:'
        if quantization_config is not None:
            training_config_text += f'\n{quantization_prefix}\n'
            training_config_text += '\n'.join([f'- {name}: {value}' for name, value in quantization_config.items()])
            training_config_text += '\n'
        training_procedure_heading = '## Training procedure'
        if quantization_prefix not in lines and bool(training_config_text):
            if training_procedure_heading in lines:
                lines.insert(lines.index(training_procedure_heading) + 2, training_config_text)
            else:
                lines.append(f'{training_procedure_heading}\n{training_config_text}')
        framework_block_heading = '### Framework versions'
        if f'- PEFT {__version__}' not in lines:
            if framework_block_heading in lines:
                lines.insert(lines.index(framework_block_heading) + 2, f'- PEFT {__version__}')
            else:
                lines.append(f'{framework_block_heading}\n\n- PEFT {__version__}')
        card.text = '\n'.join(lines)
        card.save(filename)