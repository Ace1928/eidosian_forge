import enum
from dataclasses import dataclass, field
from typing import Optional, Union
from peft.config import PromptLearningConfig
from peft.utils import PeftType
@dataclass
class PromptTuningConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`PromptEmbedding`].

    Args:
        prompt_tuning_init (Union[[`PromptTuningInit`], `str`]): The initialization of the prompt embedding.
        prompt_tuning_init_text (`str`, *optional*):
            The text to initialize the prompt embedding. Only used if `prompt_tuning_init` is `TEXT`.
        tokenizer_name_or_path (`str`, *optional*):
            The name or path of the tokenizer. Only used if `prompt_tuning_init` is `TEXT`.
        tokenizer_kwargs (`dict`, *optional*):
            The keyword arguments to pass to `AutoTokenizer.from_pretrained`. Only used if `prompt_tuning_init` is
            `TEXT`.
    """
    prompt_tuning_init: Union[PromptTuningInit, str] = field(default=PromptTuningInit.RANDOM, metadata={'help': 'How to initialize the prompt tuning parameters'})
    prompt_tuning_init_text: Optional[str] = field(default=None, metadata={'help': 'The text to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`'})
    tokenizer_name_or_path: Optional[str] = field(default=None, metadata={'help': 'The tokenizer to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`'})
    tokenizer_kwargs: Optional[dict] = field(default=None, metadata={'help': 'The keyword arguments to pass to `AutoTokenizer.from_pretrained`. Only used if prompt_tuning_init is `TEXT`'})

    def __post_init__(self):
        self.peft_type = PeftType.PROMPT_TUNING
        if self.tokenizer_kwargs and self.prompt_tuning_init != PromptTuningInit.TEXT:
            raise ValueError(f"tokenizer_kwargs only valid when using prompt_tuning_init='{PromptTuningInit.TEXT.value}'.")