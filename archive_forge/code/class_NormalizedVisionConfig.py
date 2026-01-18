import functools
from typing import Callable, Dict, Type, Union
from transformers import PretrainedConfig
class NormalizedVisionConfig(NormalizedConfig):
    IMAGE_SIZE = 'image_size'
    NUM_CHANNELS = 'num_channels'
    INPUT_SIZE = 'input_size'