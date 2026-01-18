import copy
import importlib
import json
import os
import warnings
from collections import OrderedDict
from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...utils import (
from .configuration_auto import AutoConfig, model_type_to_module_name, replace_list_option_in_docstrings
class _BaseAutoModelClass:
    _model_mapping = None

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(f'{self.__class__.__name__} is designed to be instantiated using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or `{self.__class__.__name__}.from_config(config)` methods.')

    @classmethod
    def from_config(cls, config, **kwargs):
        trust_remote_code = kwargs.pop('trust_remote_code', None)
        has_remote_code = hasattr(config, 'auto_map') and cls.__name__ in config.auto_map
        has_local_code = type(config) in cls._model_mapping.keys()
        trust_remote_code = resolve_trust_remote_code(trust_remote_code, config._name_or_path, has_local_code, has_remote_code)
        if has_remote_code and trust_remote_code:
            class_ref = config.auto_map[cls.__name__]
            if '--' in class_ref:
                repo_id, class_ref = class_ref.split('--')
            else:
                repo_id = config.name_or_path
            model_class = get_class_from_dynamic_module(class_ref, repo_id, **kwargs)
            if os.path.isdir(config._name_or_path):
                model_class.register_for_auto_class(cls.__name__)
            else:
                cls.register(config.__class__, model_class, exist_ok=True)
            _ = kwargs.pop('code_revision', None)
            return model_class._from_config(config, **kwargs)
        elif type(config) in cls._model_mapping.keys():
            model_class = _get_model_class(config, cls._model_mapping)
            return model_class._from_config(config, **kwargs)
        raise ValueError(f'Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\nModel type should be one of {', '.join((c.__name__ for c in cls._model_mapping.keys()))}.')

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop('config', None)
        trust_remote_code = kwargs.pop('trust_remote_code', None)
        kwargs['_from_auto'] = True
        hub_kwargs_names = ['cache_dir', 'force_download', 'local_files_only', 'proxies', 'resume_download', 'revision', 'subfolder', 'use_auth_token', 'token']
        hub_kwargs = {name: kwargs.pop(name) for name in hub_kwargs_names if name in kwargs}
        code_revision = kwargs.pop('code_revision', None)
        commit_hash = kwargs.pop('_commit_hash', None)
        adapter_kwargs = kwargs.pop('adapter_kwargs', None)
        token = hub_kwargs.pop('token', None)
        use_auth_token = hub_kwargs.pop('use_auth_token', None)
        if use_auth_token is not None:
            warnings.warn('The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.', FutureWarning)
            if token is not None:
                raise ValueError('`token` and `use_auth_token` are both specified. Please set only the argument `token`.')
            token = use_auth_token
        if token is not None:
            hub_kwargs['token'] = token
        if commit_hash is None:
            if not isinstance(config, PretrainedConfig):
                resolved_config_file = cached_file(pretrained_model_name_or_path, CONFIG_NAME, _raise_exceptions_for_gated_repo=False, _raise_exceptions_for_missing_entries=False, _raise_exceptions_for_connection_errors=False, **hub_kwargs)
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            else:
                commit_hash = getattr(config, '_commit_hash', None)
        if is_peft_available():
            if adapter_kwargs is None:
                adapter_kwargs = {}
                if token is not None:
                    adapter_kwargs['token'] = token
            maybe_adapter_path = find_adapter_config_file(pretrained_model_name_or_path, _commit_hash=commit_hash, **adapter_kwargs)
            if maybe_adapter_path is not None:
                with open(maybe_adapter_path, 'r', encoding='utf-8') as f:
                    adapter_config = json.load(f)
                    adapter_kwargs['_adapter_model_path'] = pretrained_model_name_or_path
                    pretrained_model_name_or_path = adapter_config['base_model_name_or_path']
        if not isinstance(config, PretrainedConfig):
            kwargs_orig = copy.deepcopy(kwargs)
            if kwargs.get('torch_dtype', None) == 'auto':
                _ = kwargs.pop('torch_dtype')
            if kwargs.get('quantization_config', None) is not None:
                _ = kwargs.pop('quantization_config')
            config, kwargs = AutoConfig.from_pretrained(pretrained_model_name_or_path, return_unused_kwargs=True, trust_remote_code=trust_remote_code, code_revision=code_revision, _commit_hash=commit_hash, **hub_kwargs, **kwargs)
            if kwargs_orig.get('torch_dtype', None) == 'auto':
                kwargs['torch_dtype'] = 'auto'
            if kwargs_orig.get('quantization_config', None) is not None:
                kwargs['quantization_config'] = kwargs_orig['quantization_config']
        has_remote_code = hasattr(config, 'auto_map') and cls.__name__ in config.auto_map
        has_local_code = type(config) in cls._model_mapping.keys()
        trust_remote_code = resolve_trust_remote_code(trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code)
        kwargs['adapter_kwargs'] = adapter_kwargs
        if has_remote_code and trust_remote_code:
            class_ref = config.auto_map[cls.__name__]
            model_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path, code_revision=code_revision, **hub_kwargs, **kwargs)
            _ = hub_kwargs.pop('code_revision', None)
            if os.path.isdir(pretrained_model_name_or_path):
                model_class.register_for_auto_class(cls.__name__)
            else:
                cls.register(config.__class__, model_class, exist_ok=True)
            return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs)
        elif type(config) in cls._model_mapping.keys():
            model_class = _get_model_class(config, cls._model_mapping)
            return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs)
        raise ValueError(f'Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\nModel type should be one of {', '.join((c.__name__ for c in cls._model_mapping.keys()))}.')

    @classmethod
    def register(cls, config_class, model_class, exist_ok=False):
        """
        Register a new model for this class.

        Args:
            config_class ([`PretrainedConfig`]):
                The configuration corresponding to the model to register.
            model_class ([`PreTrainedModel`]):
                The model to register.
        """
        if hasattr(model_class, 'config_class') and model_class.config_class != config_class:
            raise ValueError(f'The model class you are passing has a `config_class` attribute that is not consistent with the config class you passed (model has {model_class.config_class} and you passed {config_class}. Fix one of those so they match!')
        cls._model_mapping.register(config_class, model_class, exist_ok=exist_ok)