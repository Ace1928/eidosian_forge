import os
from typing import TYPE_CHECKING, Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging
class ClvpEncoderConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`ClvpEncoder`]. It is used to instantiate a CLVP
    text or CLVP speech encoder according to the specified arguments. Instantiating a configuration with the defaults
    will yield a similar configuration to that of the encoder of the CLVP
    [susnato/clvp_dev](https://huggingface.co/susnato/clvp_dev) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 256):
            Vocabulary size of the CLVP Encoder model.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 1536):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        projection_dim (`int`, *optional*, defaults to 768):
            Dimensionality of the projection vector.
        num_hidden_layers (`int`, *optional*, defaults to 20):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the feed-forward layers in [`ClvpEncoderMLP`].
        use_rotary_embedding (`bool`, *optional*, defaults to `True`):
            Whether to use rotary_embedding or not.
        use_attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in Query, Key and Value layers during self attention.
        summary_type (`str`, *optional*, defaults to `"mean"`):
            What strategy to use to get pooler_output from the last_hidden_state. `"last"`, `"first"`, `"mean"` and
            `"cls_index"` are supported.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1.0, used internally for initialization
            testing).
        bos_token_id (`int`, *optional*, defaults to 255):
            Beginning of sequence token id.
        eos_token_id (`int`, *optional*, defaults to 0):
            End of sequence token id.

    Example:

    ```python
    >>> from transformers import ClvpEncoderConfig, ClvpEncoder

    >>> # Initializing a ClvpEncoderConfig with susnato/clvp_dev style configuration
    >>> encoder_configuration = ClvpEncoderConfig()

    >>> # Initializing a ClvpEncoder (with random weights) from the susnato/clvp_dev style configuration
    >>> model = ClvpEncoder(encoder_configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = 'clvp_encoder'

    def __init__(self, vocab_size=256, hidden_size=768, intermediate_size=1536, projection_dim=768, num_hidden_layers=20, num_attention_heads=12, hidden_act='gelu', layer_norm_eps=1e-05, attention_dropout=0.1, dropout=0.1, use_rotary_embedding=True, use_attention_bias=False, summary_type='mean', initializer_factor=1.0, bos_token_id=255, eos_token_id=0, **kwargs):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.use_rotary_embedding = use_rotary_embedding
        self.use_attention_bias = use_attention_bias
        self.summary_type = summary_type
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], config_type: str='text_config', **kwargs) -> 'PretrainedConfig':
        cls._set_token_in_kwargs(kwargs)
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_type not in ['text_config', 'speech_config']:
            raise ValueError(f"We can only load either 'text_config' or 'speech_config' but you are trying to load{config_type}")
        if config_dict.get('model_type') == 'clvp':
            config_dict = config_dict[config_type]
        if 'model_type' in config_dict and hasattr(cls, 'model_type') and (config_dict['model_type'] != cls.model_type):
            logger.warning(f'You are using a model of type {config_dict['model_type']} to instantiate a model of type {cls.model_type}. This is not supported for all configurations of models and can yield errors.')
        return cls.from_dict(config_dict, **kwargs)