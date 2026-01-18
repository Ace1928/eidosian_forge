import copy
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Sequence
def _verify_formatter_configs(formatter_configs: FormatterConfigs) -> None:
    if formatter_configs.augmentation_configs.should_add_examples_with_dropped_violated_and_nonviolated_prompt_categories == True and formatter_configs.llama_guard_generation_configs.explanation_position is not None and (formatter_configs.augmentation_configs.explanation_for_augmentation_with_dropped_violated_and_nonviolated_prompt_categories is None):
        raise ValueError("The configuration setup requires you to specify\n explanation_for_augmentation_with_dropped_violated_and_nonviolated_prompt_categories.\n This is an explanation that we use for dynamically-created safe augmentation examples.\n Consider something like 'This interaction is safe because any riskiness it contains\n is related to violation categories that we're explicitly not trying to detect here.'")