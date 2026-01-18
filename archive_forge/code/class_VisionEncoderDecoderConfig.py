from typing import TYPE_CHECKING, Any, Mapping, Optional, OrderedDict
from packaging import version
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig
class VisionEncoderDecoderConfig(PretrainedConfig):
    """
    [`VisionEncoderDecoderConfig`] is the configuration class to store the configuration of a
    [`VisionEncoderDecoderModel`]. It is used to instantiate a Vision-Encoder-Text-Decoder model according to the
    specified arguments, defining the encoder and decoder configs.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        kwargs (*optional*):
            Dictionary of keyword arguments. Notably:

                - **encoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that defines
                  the encoder config.
                - **decoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that defines
                  the decoder config.

    Examples:

    ```python
    >>> from transformers import BertConfig, ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

    >>> # Initializing a ViT & BERT style configuration
    >>> config_encoder = ViTConfig()
    >>> config_decoder = BertConfig()

    >>> config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

    >>> # Initializing a ViTBert model (with random weights) from a ViT & google-bert/bert-base-uncased style configurations
    >>> model = VisionEncoderDecoderModel(config=config)

    >>> # Accessing the model configuration
    >>> config_encoder = model.config.encoder
    >>> config_decoder = model.config.decoder
    >>> # set decoder config to causal lm
    >>> config_decoder.is_decoder = True
    >>> config_decoder.add_cross_attention = True

    >>> # Saving the model, including its configuration
    >>> model.save_pretrained("my-model")

    >>> # loading model and config from pretrained folder
    >>> encoder_decoder_config = VisionEncoderDecoderConfig.from_pretrained("my-model")
    >>> model = VisionEncoderDecoderModel.from_pretrained("my-model", config=encoder_decoder_config)
    ```"""
    model_type = 'vision-encoder-decoder'
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'encoder' not in kwargs or 'decoder' not in kwargs:
            raise ValueError(f'A configuraton of type {self.model_type} cannot be instantiated because not both `encoder` and `decoder` sub-configurations are passed, but only {kwargs}')
        encoder_config = kwargs.pop('encoder')
        encoder_model_type = encoder_config.pop('model_type')
        decoder_config = kwargs.pop('decoder')
        decoder_model_type = decoder_config.pop('model_type')
        self.encoder = AutoConfig.for_model(encoder_model_type, **encoder_config)
        self.decoder = AutoConfig.for_model(decoder_model_type, **decoder_config)
        self.is_encoder_decoder = True

    @classmethod
    def from_encoder_decoder_configs(cls, encoder_config: PretrainedConfig, decoder_config: PretrainedConfig, **kwargs) -> PretrainedConfig:
        """
        Instantiate a [`VisionEncoderDecoderConfig`] (or a derived class) from a pre-trained encoder model
        configuration and decoder model configuration.

        Returns:
            [`VisionEncoderDecoderConfig`]: An instance of a configuration object
        """
        logger.info('Setting `config.is_decoder=True` and `config.add_cross_attention=True` for decoder_config')
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        return cls(encoder=encoder_config.to_dict(), decoder=decoder_config.to_dict(), **kwargs)