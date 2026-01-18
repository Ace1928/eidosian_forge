import inspect
import re
import string
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
class SupportLevel(Enum):
    """
    Indicates at what stage the feature
    used in the example is handled in export.
    """
    SUPPORTED = 1
    NOT_SUPPORTED_YET = 0