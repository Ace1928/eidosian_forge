import copy
import enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from vllm.block import LogicalTokenBlock
from vllm.prefix import Prefix
from vllm.sampling_params import SamplingParams
from vllm.lora.request import LoRARequest
class SequenceGroupOutput:
    """The model output associated with a sequence group."""

    def __init__(self, samples: List[SequenceOutput], prompt_logprobs: Optional[PromptLogprobs]) -> None:
        self.samples = samples
        self.prompt_logprobs = prompt_logprobs

    def __repr__(self) -> str:
        return f'SequenceGroupOutput(samples={self.samples}, prompt_logprobs={self.prompt_logprobs})'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequenceGroupOutput):
            raise NotImplementedError()
        return self.samples == other.samples and self.prompt_logprobs == other.prompt_logprobs