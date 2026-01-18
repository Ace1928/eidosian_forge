import os
import warnings
from dataclasses import asdict
from enum import Enum
from typing import List, Optional, Union
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import is_tf_available, logging
from .utils import DataProcessor, InputExample, InputFeatures
class OutputMode(Enum):
    classification = 'classification'
    regression = 'regression'