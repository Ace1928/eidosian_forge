import copy
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Sequence
def _convert_category_codes_to_indices(codes: List[str], formatter_configs: FormatterConfigs) -> List[int]:
    return [int(code.lstrip(formatter_configs.guidelines.category_code_prefix)) - 1 for code in codes]