from typing import Dict
from ...configuration_utils import PretrainedConfig
from ...utils import logging
class FastSpeech2ConformerHifiGanConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`FastSpeech2ConformerHifiGanModel`]. It is used to
    instantiate a FastSpeech2Conformer HiFi-GAN vocoder model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    FastSpeech2Conformer
    [espnet/fastspeech2_conformer_hifigan](https://huggingface.co/espnet/fastspeech2_conformer_hifigan) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        model_in_dim (`int`, *optional*, defaults to 80):
            The number of frequency bins in the input log-mel spectrogram.
        upsample_initial_channel (`int`, *optional*, defaults to 512):
            The number of input channels into the upsampling network.
        upsample_rates (`Tuple[int]` or `List[int]`, *optional*, defaults to `[8, 8, 2, 2]`):
            A tuple of integers defining the stride of each 1D convolutional layer in the upsampling network. The
            length of *upsample_rates* defines the number of convolutional layers and has to match the length of
            *upsample_kernel_sizes*.
        upsample_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[16, 16, 4, 4]`):
            A tuple of integers defining the kernel size of each 1D convolutional layer in the upsampling network. The
            length of *upsample_kernel_sizes* defines the number of convolutional layers and has to match the length of
            *upsample_rates*.
        resblock_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[3, 7, 11]`):
            A tuple of integers defining the kernel sizes of the 1D convolutional layers in the multi-receptive field
            fusion (MRF) module.
        resblock_dilation_sizes (`Tuple[Tuple[int]]` or `List[List[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`):
            A nested tuple of integers defining the dilation rates of the dilated 1D convolutional layers in the
            multi-receptive field fusion (MRF) module.
        initializer_range (`float`, *optional*, defaults to 0.01):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        leaky_relu_slope (`float`, *optional*, defaults to 0.1):
            The angle of the negative slope used by the leaky ReLU activation.
        normalize_before (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the spectrogram before vocoding using the vocoder's learned mean and variance.

    Example:

    ```python
    >>> from transformers import FastSpeech2ConformerHifiGan, FastSpeech2ConformerHifiGanConfig

    >>> # Initializing a FastSpeech2ConformerHifiGan configuration
    >>> configuration = FastSpeech2ConformerHifiGanConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = FastSpeech2ConformerHifiGan(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = 'hifigan'

    def __init__(self, model_in_dim=80, upsample_initial_channel=512, upsample_rates=[8, 8, 2, 2], upsample_kernel_sizes=[16, 16, 4, 4], resblock_kernel_sizes=[3, 7, 11], resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]], initializer_range=0.01, leaky_relu_slope=0.1, normalize_before=True, **kwargs):
        self.model_in_dim = model_in_dim
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.initializer_range = initializer_range
        self.leaky_relu_slope = leaky_relu_slope
        self.normalize_before = normalize_before
        super().__init__(**kwargs)