import copy
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Sequence
def _get_map_of_original_category_indices_to_rewritten_category_codes(formatter_configs: FormatterConfigs, category_indices_included_in_llama_guard_prompt: List[int]) -> Dict[int, str]:
    to_return = {}
    for rewritten_category_index, original_category_index in enumerate(category_indices_included_in_llama_guard_prompt):
        to_return[original_category_index] = formatter_configs.guidelines.category_code_prefix + str(rewritten_category_index + 1)
    return to_return