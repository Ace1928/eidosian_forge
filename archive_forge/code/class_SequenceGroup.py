import copy
import enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from vllm.block import LogicalTokenBlock
from vllm.prefix import Prefix
from vllm.sampling_params import SamplingParams
from vllm.lora.request import LoRARequest
class SequenceGroup:
    """A group of sequences that are generated from the same prompt.

    Args:
        request_id: The ID of the request.
        seqs: The list of sequences.
        sampling_params: The sampling parameters used to generate the outputs.
        arrival_time: The arrival time of the request.
        lora_request: LoRA request.
        prefix: The prefix of the prompt of the sequence group.
    """

    def __init__(self, request_id: str, seqs: List[Sequence], sampling_params: SamplingParams, arrival_time: float, lora_request: Optional[LoRARequest]=None, prefix: Optional[Prefix]=None) -> None:
        self.request_id = request_id
        self.seqs_dict = {seq.seq_id: seq for seq in seqs}
        self.sampling_params = sampling_params
        self.metrics = RequestMetrics(arrival_time=arrival_time, last_token_time=arrival_time, first_scheduled_time=None, first_token_time=None, time_in_queue=None)
        self.lora_request = lora_request
        self.prefix: Optional[Prefix] = prefix
        self.prompt_logprobs: Optional[PromptLogprobs] = None
        self.state = SequenceGroupState()

    @property
    def prompt(self) -> str:
        return next(iter(self.seqs_dict.values())).prompt

    @property
    def prompt_token_ids(self) -> List[int]:
        return next(iter(self.seqs_dict.values())).data.prompt_token_ids

    @property
    def lora_int_id(self) -> int:
        return self.lora_request.lora_int_id if self.lora_request else 0

    def get_last_latency(self, now: float) -> float:
        """Gets last token latency for Request level timings."""
        latency = now - self.metrics.last_token_time
        self.metrics.last_token_time = now
        return latency

    def maybe_set_first_token_time(self, time: float) -> None:
        """Sets the first token time for Request level timings."""
        if self.metrics.first_token_time is None:
            self.metrics.first_token_time = time

    def maybe_set_first_scheduled_time(self, time: float) -> None:
        """Sets the first scheduled time and time in queue for Request level timings."""
        if self.metrics.first_scheduled_time is None:
            self.metrics.first_scheduled_time = time
            self.metrics.time_in_queue = time - self.metrics.arrival_time

    def set_finished_time(self, time: Optional[float]) -> None:
        """Sets the finished time for Request level timings."""
        self.metrics.finished_time = time

    def get_max_num_running_seqs(self) -> int:
        """The maximum number of sequences running in parallel in the remaining
        lifetime of the request."""
        if self.sampling_params.use_beam_search:
            return self.sampling_params.best_of
        else:
            if self.sampling_params.best_of > self.num_seqs():
                return self.sampling_params.best_of
            return self.num_unfinished_seqs()

    def get_seqs(self, status: Optional[SequenceStatus]=None) -> List[Sequence]:
        if status is None:
            return list(self.seqs_dict.values())
        else:
            return [seq for seq in self.seqs_dict.values() if seq.status == status]

    def get_unfinished_seqs(self) -> List[Sequence]:
        return [seq for seq in self.seqs_dict.values() if not seq.is_finished()]

    def get_finished_seqs(self) -> List[Sequence]:
        return [seq for seq in self.seqs_dict.values() if seq.is_finished()]

    def num_seqs(self, status: Optional[SequenceStatus]=None) -> int:
        return len(self.get_seqs(status))

    def num_unfinished_seqs(self) -> int:
        return len(self.get_unfinished_seqs())

    def num_finished_seqs(self) -> int:
        return len(self.get_finished_seqs())

    def find(self, seq_id: int) -> Sequence:
        if seq_id not in self.seqs_dict:
            raise ValueError(f'Sequence {seq_id} not found.')
        return self.seqs_dict[seq_id]

    def add(self, seq: Sequence) -> None:
        if seq.seq_id in self.seqs_dict:
            raise ValueError(f'Sequence {seq.seq_id} already exists.')
        self.seqs_dict[seq.seq_id] = seq

    def remove(self, seq_id: int) -> None:
        if seq_id not in self.seqs_dict:
            raise ValueError(f'Sequence {seq_id} not found.')
        del self.seqs_dict[seq_id]

    def is_finished(self) -> bool:
        return all((seq.is_finished() for seq in self.get_seqs()))

    def __repr__(self) -> str:
        return f'SequenceGroup(request_id={self.request_id}, sampling_params={self.sampling_params}, num_seqs={len(self.seqs_dict)})'