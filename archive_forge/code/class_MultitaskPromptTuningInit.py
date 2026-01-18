import enum
from dataclasses import dataclass, field
from typing import Optional, Union
from peft.tuners.prompt_tuning import PromptTuningConfig
from peft.utils import PeftType
class MultitaskPromptTuningInit(str, enum.Enum):
    TEXT = 'TEXT'
    RANDOM = 'RANDOM'
    AVERAGE_SOURCE_TASKS = 'AVERAGE_SOURCE_TASKS'
    EXACT_SOURCE_TASK = 'EXACT_SOURCE_TASK'
    ONLY_SOURCE_SHARED = 'ONLY_SOURCE_SHARED'