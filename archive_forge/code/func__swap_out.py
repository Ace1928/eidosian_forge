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
def _swap_out(self, seq_group: SequenceGroup, blocks_to_swap_out: Dict[int, int]) -> None:
    if not self.block_manager.can_swap_out(seq_group):
        raise RuntimeError('Aborted due to the lack of CPU swap space. Please increase the swap space to avoid this error.')
    mapping = self.block_manager.swap_out(seq_group)
    blocks_to_swap_out.update(mapping)
    for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
        seq.status = SequenceStatus.SWAPPED