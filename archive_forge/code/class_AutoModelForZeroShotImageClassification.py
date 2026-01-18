import warnings
from collections import OrderedDict
from ...utils import logging
from .auto_factory import (
from .configuration_auto import CONFIG_MAPPING_NAMES
class AutoModelForZeroShotImageClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING