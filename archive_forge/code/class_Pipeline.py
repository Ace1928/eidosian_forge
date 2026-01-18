import collections
import csv
import importlib
import json
import os
import pickle
import sys
import traceback
import types
import warnings
from abc import ABC, abstractmethod
from collections import UserDict
from contextlib import contextmanager
from os.path import abspath, exists
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from ..dynamic_module_utils import custom_object_save
from ..feature_extraction_utils import PreTrainedFeatureExtractor
from ..image_processing_utils import BaseImageProcessor
from ..modelcard import ModelCard
from ..models.auto.configuration_auto import AutoConfig
from ..tokenization_utils import PreTrainedTokenizer
from ..utils import (
@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True, has_feature_extractor=True, has_image_processor=True))
class Pipeline(_ScikitCompat):
    """
    The Pipeline class is the class from which all pipelines inherit. Refer to this class for methods shared across
    different pipelines.

    Base class implementing pipelined operations. Pipeline workflow is defined as a sequence of the following
    operations:

        Input -> Tokenization -> Model Inference -> Post-Processing (task dependent) -> Output

    Pipeline supports running on CPU or GPU through the device argument (see below).

    Some pipeline, like for instance [`FeatureExtractionPipeline`] (`'feature-extraction'`) output large tensor object
    as nested-lists. In order to avoid dumping such large structure as textual data we provide the `binary_output`
    constructor argument. If set to `True`, the output will be stored in the pickle format.
    """
    default_input_names = None

    def __init__(self, model: Union['PreTrainedModel', 'TFPreTrainedModel'], tokenizer: Optional[PreTrainedTokenizer]=None, feature_extractor: Optional[PreTrainedFeatureExtractor]=None, image_processor: Optional[BaseImageProcessor]=None, modelcard: Optional[ModelCard]=None, framework: Optional[str]=None, task: str='', args_parser: ArgumentHandler=None, device: Union[int, 'torch.device']=None, torch_dtype: Optional[Union[str, 'torch.dtype']]=None, binary_output: bool=False, **kwargs):
        if framework is None:
            framework, model = infer_framework_load_model(model, config=model.config)
        self.task = task
        self.model = model
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.image_processor = image_processor
        self.modelcard = modelcard
        self.framework = framework
        hf_device_map = getattr(self.model, 'hf_device_map', None)
        if hf_device_map is not None and device is not None:
            raise ValueError('The model has been loaded with `accelerate` and therefore cannot be moved to a specific device. Please discard the `device` argument when creating your pipeline object.')
        if device is None:
            if hf_device_map is not None:
                device = next(iter(hf_device_map.values()))
            else:
                device = -1
        if is_torch_available() and self.framework == 'pt':
            if isinstance(device, torch.device):
                if device.type == 'xpu' and (not is_torch_xpu_available(check_device=True)):
                    raise ValueError(f'{device} is not available, you should use device="cpu" instead')
                self.device = device
            elif isinstance(device, str):
                if 'xpu' in device and (not is_torch_xpu_available(check_device=True)):
                    raise ValueError(f'{device} is not available, you should use device="cpu" instead')
                self.device = torch.device(device)
            elif device < 0:
                self.device = torch.device('cpu')
            elif is_torch_cuda_available():
                self.device = torch.device(f'cuda:{device}')
            elif is_torch_npu_available():
                self.device = torch.device(f'npu:{device}')
            elif is_torch_xpu_available(check_device=True):
                self.device = torch.device(f'xpu:{device}')
            else:
                raise ValueError(f'{device} unrecognized or not available.')
        else:
            self.device = device if device is not None else -1
        self.torch_dtype = torch_dtype
        self.binary_output = binary_output
        if self.framework == 'pt' and self.device is not None and (not (isinstance(self.device, int) and self.device < 0)) and (hf_device_map is None):
            self.model.to(self.device)
        task_specific_params = self.model.config.task_specific_params
        if task_specific_params is not None and task in task_specific_params:
            self.model.config.update(task_specific_params.get(task))
            if self.model.can_generate():
                self.model.generation_config.update(**task_specific_params.get(task))
        self.call_count = 0
        self._batch_size = kwargs.pop('batch_size', None)
        self._num_workers = kwargs.pop('num_workers', None)
        self._preprocess_params, self._forward_params, self._postprocess_params = self._sanitize_parameters(**kwargs)
        if self.image_processor is None and self.feature_extractor is not None:
            if isinstance(self.feature_extractor, BaseImageProcessor):
                self.image_processor = self.feature_extractor

    def save_pretrained(self, save_directory: str, safe_serialization: bool=True):
        """
        Save the pipeline's model and tokenizer.

        Args:
            save_directory (`str`):
                A path to the directory where to saved. It will be created if it doesn't exist.
            safe_serialization (`str`):
                Whether to save the model using `safetensors` or the traditional way for PyTorch or Tensorflow.
        """
        if os.path.isfile(save_directory):
            logger.error(f'Provided path ({save_directory}) should be a directory, not a file')
            return
        os.makedirs(save_directory, exist_ok=True)
        if hasattr(self, '_registered_impl'):
            pipeline_info = self._registered_impl.copy()
            custom_pipelines = {}
            for task, info in pipeline_info.items():
                if info['impl'] != self.__class__:
                    continue
                info = info.copy()
                module_name = info['impl'].__module__
                last_module = module_name.split('.')[-1]
                info['impl'] = f'{last_module}.{info['impl'].__name__}'
                info['pt'] = tuple((c.__name__ for c in info['pt']))
                info['tf'] = tuple((c.__name__ for c in info['tf']))
                custom_pipelines[task] = info
            self.model.config.custom_pipelines = custom_pipelines
            custom_object_save(self, save_directory)
        self.model.save_pretrained(save_directory, safe_serialization=safe_serialization)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)
        if self.feature_extractor is not None:
            self.feature_extractor.save_pretrained(save_directory)
        if self.image_processor is not None:
            self.image_processor.save_pretrained(save_directory)
        if self.modelcard is not None:
            self.modelcard.save_pretrained(save_directory)

    def transform(self, X):
        """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        """
        return self(X)

    def predict(self, X):
        """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        """
        return self(X)

    @contextmanager
    def device_placement(self):
        """
        Context Manager allowing tensor allocation on the user-specified device in framework agnostic way.

        Returns:
            Context manager

        Examples:

        ```python
        # Explicitly ask for tensor allocation on CUDA device :0
        pipe = pipeline(..., device=0)
        with pipe.device_placement():
            # Every framework specific tensor allocation will be done on the request device
            output = pipe(...)
        ```"""
        if self.framework == 'tf':
            with tf.device('/CPU:0' if self.device == -1 else f'/device:GPU:{self.device}'):
                yield
        elif self.device.type == 'cuda':
            with torch.cuda.device(self.device):
                yield
        else:
            yield

    def ensure_tensor_on_device(self, **inputs):
        """
        Ensure PyTorch tensors are on the specified device.

        Args:
            inputs (keyword arguments that should be `torch.Tensor`, the rest is ignored):
                The tensors to place on `self.device`.
            Recursive on lists **only**.

        Return:
            `Dict[str, torch.Tensor]`: The same as `inputs` but on the proper device.
        """
        return self._ensure_tensor_on_device(inputs, self.device)

    def _ensure_tensor_on_device(self, inputs, device):
        if isinstance(inputs, ModelOutput):
            return ModelOutput({name: self._ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()})
        elif isinstance(inputs, dict):
            return {name: self._ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()}
        elif isinstance(inputs, UserDict):
            return UserDict({name: self._ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()})
        elif isinstance(inputs, list):
            return [self._ensure_tensor_on_device(item, device) for item in inputs]
        elif isinstance(inputs, tuple):
            return tuple([self._ensure_tensor_on_device(item, device) for item in inputs])
        elif isinstance(inputs, torch.Tensor):
            if device == torch.device('cpu') and inputs.dtype in {torch.float16, torch.bfloat16}:
                inputs = inputs.float()
            return inputs.to(device)
        else:
            return inputs

    def check_model_type(self, supported_models: Union[List[str], dict]):
        """
        Check if the model class is in supported by the pipeline.

        Args:
            supported_models (`List[str]` or `dict`):
                The list of models supported by the pipeline, or a dictionary with model class values.
        """
        if not isinstance(supported_models, list):
            supported_models_names = []
            for _, model_name in supported_models.items():
                if isinstance(model_name, tuple):
                    supported_models_names.extend(list(model_name))
                else:
                    supported_models_names.append(model_name)
            if hasattr(supported_models, '_model_mapping'):
                for _, model in supported_models._model_mapping._extra_content.items():
                    if isinstance(model_name, tuple):
                        supported_models_names.extend([m.__name__ for m in model])
                    else:
                        supported_models_names.append(model.__name__)
            supported_models = supported_models_names
        if self.model.__class__.__name__ not in supported_models:
            logger.error(f"The model '{self.model.__class__.__name__}' is not supported for {self.task}. Supported models are {supported_models}.")

    @abstractmethod
    def _sanitize_parameters(self, **pipeline_parameters):
        """
        _sanitize_parameters will be called with any excessive named arguments from either `__init__` or `__call__`
        methods. It should return 3 dictionaries of the resolved parameters used by the various `preprocess`,
        `forward` and `postprocess` methods. Do not fill dictionaries if the caller didn't specify a kwargs. This
        lets you keep defaults in function signatures, which is more "natural".

        It is not meant to be called directly, it will be automatically called and the final parameters resolved by
        `__init__` and `__call__`
        """
        raise NotImplementedError('_sanitize_parameters not implemented')

    @abstractmethod
    def preprocess(self, input_: Any, **preprocess_parameters: Dict) -> Dict[str, GenericTensor]:
        """
        Preprocess will take the `input_` of a specific pipeline and return a dictionary of everything necessary for
        `_forward` to run properly. It should contain at least one tensor, but might have arbitrary other items.
        """
        raise NotImplementedError('preprocess not implemented')

    @abstractmethod
    def _forward(self, input_tensors: Dict[str, GenericTensor], **forward_parameters: Dict) -> ModelOutput:
        """
        _forward will receive the prepared dictionary from `preprocess` and run it on the model. This method might
        involve the GPU or the CPU and should be agnostic to it. Isolating this function is the reason for `preprocess`
        and `postprocess` to exist, so that the hot path, this method generally can run as fast as possible.

        It is not meant to be called directly, `forward` is preferred. It is basically the same but contains additional
        code surrounding `_forward` making sure tensors and models are on the same device, disabling the training part
        of the code (leading to faster inference).
        """
        raise NotImplementedError('_forward not implemented')

    @abstractmethod
    def postprocess(self, model_outputs: ModelOutput, **postprocess_parameters: Dict) -> Any:
        """
        Postprocess will receive the raw outputs of the `_forward` method, generally tensors, and reformat them into
        something more friendly. Generally it will output a list or a dict or results (containing just strings and
        numbers).
        """
        raise NotImplementedError('postprocess not implemented')

    def get_inference_context(self):
        return torch.no_grad

    def forward(self, model_inputs, **forward_params):
        with self.device_placement():
            if self.framework == 'tf':
                model_inputs['training'] = False
                model_outputs = self._forward(model_inputs, **forward_params)
            elif self.framework == 'pt':
                inference_context = self.get_inference_context()
                with inference_context():
                    model_inputs = self._ensure_tensor_on_device(model_inputs, device=self.device)
                    model_outputs = self._forward(model_inputs, **forward_params)
                    model_outputs = self._ensure_tensor_on_device(model_outputs, device=torch.device('cpu'))
            else:
                raise ValueError(f'Framework {self.framework} is not supported')
        return model_outputs

    def get_iterator(self, inputs, num_workers: int, batch_size: int, preprocess_params, forward_params, postprocess_params):
        if isinstance(inputs, collections.abc.Sized):
            dataset = PipelineDataset(inputs, self.preprocess, preprocess_params)
        else:
            if num_workers > 1:
                logger.warning('For iterable dataset using num_workers>1 is likely to result in errors since everything is iterable, setting `num_workers=1` to guarantee correctness.')
                num_workers = 1
            dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)
        if 'TOKENIZERS_PARALLELISM' not in os.environ:
            logger.info("Disabling tokenizer parallelism, we're using DataLoader multithreading already")
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        feature_extractor = self.feature_extractor if self.feature_extractor is not None else self.image_processor
        collate_fn = no_collate_fn if batch_size == 1 else pad_collate_fn(self.tokenizer, feature_extractor)
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn)
        model_iterator = PipelineIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
        return final_iterator

    def __call__(self, inputs, *args, num_workers=None, batch_size=None, **kwargs):
        if args:
            logger.warning(f'Ignoring args : {args}')
        if num_workers is None:
            if self._num_workers is None:
                num_workers = 0
            else:
                num_workers = self._num_workers
        if batch_size is None:
            if self._batch_size is None:
                batch_size = 1
            else:
                batch_size = self._batch_size
        preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(**kwargs)
        preprocess_params = {**self._preprocess_params, **preprocess_params}
        forward_params = {**self._forward_params, **forward_params}
        postprocess_params = {**self._postprocess_params, **postprocess_params}
        self.call_count += 1
        if self.call_count > 10 and self.framework == 'pt' and (self.device.type == 'cuda'):
            warnings.warn('You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset', UserWarning)
        is_dataset = Dataset is not None and isinstance(inputs, Dataset)
        is_generator = isinstance(inputs, types.GeneratorType)
        is_list = isinstance(inputs, list)
        is_iterable = is_dataset or is_generator or is_list
        can_use_iterator = self.framework == 'pt' and (is_dataset or is_generator or is_list)
        if is_list:
            if can_use_iterator:
                final_iterator = self.get_iterator(inputs, num_workers, batch_size, preprocess_params, forward_params, postprocess_params)
                outputs = list(final_iterator)
                return outputs
            else:
                return self.run_multi(inputs, preprocess_params, forward_params, postprocess_params)
        elif can_use_iterator:
            return self.get_iterator(inputs, num_workers, batch_size, preprocess_params, forward_params, postprocess_params)
        elif is_iterable:
            return self.iterate(inputs, preprocess_params, forward_params, postprocess_params)
        elif self.framework == 'pt' and isinstance(self, ChunkPipeline):
            return next(iter(self.get_iterator([inputs], num_workers, batch_size, preprocess_params, forward_params, postprocess_params)))
        else:
            return self.run_single(inputs, preprocess_params, forward_params, postprocess_params)

    def run_multi(self, inputs, preprocess_params, forward_params, postprocess_params):
        return [self.run_single(item, preprocess_params, forward_params, postprocess_params) for item in inputs]

    def run_single(self, inputs, preprocess_params, forward_params, postprocess_params):
        model_inputs = self.preprocess(inputs, **preprocess_params)
        model_outputs = self.forward(model_inputs, **forward_params)
        outputs = self.postprocess(model_outputs, **postprocess_params)
        return outputs

    def iterate(self, inputs, preprocess_params, forward_params, postprocess_params):
        for input_ in inputs:
            yield self.run_single(input_, preprocess_params, forward_params, postprocess_params)