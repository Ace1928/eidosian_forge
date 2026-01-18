from collections import deque
import enum
import time
from typing import Deque, Dict, Iterable, List, Optional, Tuple, Union, Set
from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.block_manager import AllocStatus, BlockSpaceManager
from vllm.core.policy import PolicyFactory
from vllm.lora.request import LoRARequest
from vllm.logger import init_logger
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
from vllm.prefix import PrefixPool
class SchedulerOutputs:

    def __init__(self, scheduled_seq_groups: Iterable[SequenceGroup], prompt_run: bool, num_batched_tokens: int, blocks_to_swap_in: Dict[int, int], blocks_to_swap_out: Dict[int, int], blocks_to_copy: Dict[int, List[int]], ignored_seq_groups: List[SequenceGroup]) -> None:
        self.scheduled_seq_groups = scheduled_seq_groups
        self.prompt_run = prompt_run
        self.num_batched_tokens = num_batched_tokens
        self.blocks_to_swap_in = blocks_to_swap_in
        self.blocks_to_swap_out = blocks_to_swap_out
        self.blocks_to_copy = blocks_to_copy
        assert not (blocks_to_swap_in and blocks_to_swap_out)
        self.ignored_seq_groups = ignored_seq_groups
        self.num_loras = len(self.lora_requests)
        if self.num_loras > 0:
            self._sort_by_lora_ids()

    def is_empty(self) -> bool:
        return not self.scheduled_seq_groups and (not self.blocks_to_swap_in) and (not self.blocks_to_swap_out) and (not self.blocks_to_copy)

    def _sort_by_lora_ids(self) -> bool:
        self.scheduled_seq_groups = sorted(self.scheduled_seq_groups, key=lambda g: (g.lora_request.lora_int_id if g.lora_request else 0, g.request_id))

    @property
    def lora_requests(self) -> Set[LoRARequest]:
        return {g.lora_request for g in self.scheduled_seq_groups}