from collections import OrderedDict
from ...utils import logging
from .auto_factory import _BaseAutoModelClass, _LazyAutoMapping, auto_class_update
from .configuration_auto import CONFIG_MAPPING_NAMES
class FlaxAutoModelForPreTraining(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_PRETRAINING_MAPPING