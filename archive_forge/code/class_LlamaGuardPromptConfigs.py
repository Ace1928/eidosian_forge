import copy
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Sequence
@dataclass
class LlamaGuardPromptConfigs:
    instructions_format_string: str
    should_include_category_descriptions: bool
    should_shuffle_category_codes: bool = True