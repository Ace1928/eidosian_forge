import copy
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Sequence
@dataclass
class AugmentationConfigs:
    should_add_examples_with_dropped_nonviolated_prompt_categories: bool = True
    should_add_examples_with_dropped_violated_and_nonviolated_prompt_categories: bool = False
    explanation_for_augmentation_with_dropped_violated_and_nonviolated_prompt_categories: Optional[str] = None