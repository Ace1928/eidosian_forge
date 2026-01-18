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
def _swap_in(self, seq_group: SequenceGroup, blocks_to_swap_in: Dict[int, int]) -> None:
    mapping = self.block_manager.swap_in(seq_group)
    blocks_to_swap_in.update(mapping)
    for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
        seq.status = SequenceStatus.RUNNING