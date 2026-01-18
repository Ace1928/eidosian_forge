import warnings
from collections import OrderedDict
from ...utils import logging
from .auto_factory import (
from .configuration_auto import CONFIG_MAPPING_NAMES
class AutoModelForTableQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING