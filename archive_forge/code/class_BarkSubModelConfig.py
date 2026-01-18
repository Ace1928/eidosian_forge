import os
from typing import Dict, Optional, Union
from ...configuration_utils import PretrainedConfig
from ...utils import add_start_docstrings, logging
from ..auto import CONFIG_MAPPING
class BarkSubModelConfig(PretrainedConfig):
    model_type = 'bark_module'
    keys_to_ignore_at_inference = ['past_key_values']
    attribute_map = {'num_attention_heads': 'num_heads', 'num_hidden_layers': 'num_layers', 'vocab_size': 'input_vocab_size', 'window_size': 'block_size'}

    def __init__(self, block_size=1024, input_vocab_size=10048, output_vocab_size=10048, num_layers=12, num_heads=12, hidden_size=768, dropout=0.0, bias=True, initializer_range=0.02, use_cache=True, **kwargs):
        self.block_size = block_size
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.bias = bias
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], cache_dir: Optional[Union[str, os.PathLike]]=None, force_download: bool=False, local_files_only: bool=False, token: Optional[Union[str, bool]]=None, revision: str='main', **kwargs) -> 'PretrainedConfig':
        kwargs['cache_dir'] = cache_dir
        kwargs['force_download'] = force_download
        kwargs['local_files_only'] = local_files_only
        kwargs['revision'] = revision
        cls._set_token_in_kwargs(kwargs, token)
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict.get('model_type') == 'bark':
            config_dict = config_dict[f'{cls.model_type}_config']
        if 'model_type' in config_dict and hasattr(cls, 'model_type') and (config_dict['model_type'] != cls.model_type):
            logger.warning(f'You are using a model of type {config_dict['model_type']} to instantiate a model of type {cls.model_type}. This is not supported for all configurations of models and can yield errors.')
        return cls.from_dict(config_dict, **kwargs)