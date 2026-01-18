import warnings
from collections import OrderedDict
from ...utils import logging
from .auto_factory import (
from .configuration_auto import CONFIG_MAPPING_NAMES
class AutoModelForImageSegmentation(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_IMAGE_SEGMENTATION_MAPPING