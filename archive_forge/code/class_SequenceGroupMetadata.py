import copy
import enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from vllm.block import LogicalTokenBlock
from vllm.prefix import Prefix
from vllm.sampling_params import SamplingParams
from vllm.lora.request import LoRARequest
class SequenceGroupMetadata:
    """Metadata for a sequence group. Used to create `InputMetadata`.

    Args:
        request_id: The ID of the request.
        is_prompt: Whether the request is at prompt stage.
        seq_data: The sequence data. (Seq id -> sequence data)
        sampling_params: The sampling parameters used to generate the outputs.
        block_tables: The block tables. (Seq id -> list of physical block
            numbers)
        state: Internal state tied to this sequence group.
        lora_request: LoRA request.
        prefix: The prefix of the prompt of the sequence group.
    """

    def __init__(self, request_id: str, is_prompt: bool, seq_data: Dict[int, SequenceData], sampling_params: SamplingParams, block_tables: Dict[int, List[int]], lora_request: Optional[LoRARequest]=None, prefix: Optional[Prefix]=None, state: Optional[SequenceGroupState]=None) -> None:
        self.request_id = request_id
        self.is_prompt = is_prompt
        self.seq_data = seq_data
        self.sampling_params = sampling_params
        self.block_tables = block_tables
        self.lora_request = lora_request
        self.prefix = prefix
        self.state = SequenceGroupState() if state is None else state

    @property
    def lora_int_id(self) -> int:
        return self.lora_request.lora_int_id if self.lora_request else 0