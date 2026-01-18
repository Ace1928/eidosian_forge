import warnings
from collections import OrderedDict
from ...utils import logging
from .auto_factory import (
from .configuration_auto import CONFIG_MAPPING_NAMES
class AutoBackbone(_BaseAutoBackboneClass):
    _model_mapping = MODEL_FOR_BACKBONE_MAPPING