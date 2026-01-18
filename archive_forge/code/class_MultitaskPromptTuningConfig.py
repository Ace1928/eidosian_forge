import enum
from dataclasses import dataclass, field
from typing import Optional, Union
from peft.tuners.prompt_tuning import PromptTuningConfig
from peft.utils import PeftType
@dataclass
class MultitaskPromptTuningConfig(PromptTuningConfig):
    prompt_tuning_init: Union[MultitaskPromptTuningInit, str] = field(default=MultitaskPromptTuningInit.RANDOM, metadata={'help': 'How to initialize the prompt tuning parameters. Can be one of TEXT, RANDOM, AVERAGE_SOURCE_TASKS, EXACT_SOURCE_TASK, ONLY_SOURCE_SHARED.'})
    prompt_tuning_init_state_dict_path: Optional[str] = field(default=None, metadata={'help': 'The path of source state dict. This is required when training the downstream target prompt from the pretrained source prompt'})
    prompt_tuning_init_task: Optional[int] = field(default=0, metadata={'help': 'source task id for initialization'})
    num_ranks: Optional[int] = field(default=1, metadata={'help': 'ranks'})
    num_tasks: Optional[int] = field(default=1, metadata={'help': 'number of tasks'})

    def __post_init__(self):
        self.peft_type = PeftType.MULTITASK_PROMPT_TUNING