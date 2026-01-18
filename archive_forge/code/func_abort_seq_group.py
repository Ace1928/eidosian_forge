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
def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
    """Aborts a sequence group with the given ID.

        Check if the sequence group with the given ID
            is present in any of the state queue.
        If present, remove the sequence group from the state queue.
            Also, if any of the sequences in the sequence group is not finished,
                free the sequence with status `FINISHED_ABORTED`.
        Otherwise, do nothing.

        Args:
            request_id: The ID(s) of the sequence group to abort.
        """
    if isinstance(request_id, str):
        request_id = (request_id,)
    request_ids = set(request_id)
    for state_queue in [self.waiting, self.running, self.swapped]:
        aborted_groups: List[SequenceGroup] = []
        for seq_group in state_queue:
            if not request_ids:
                break
            if seq_group.request_id in request_ids:
                aborted_groups.append(seq_group)
                request_ids.remove(seq_group.request_id)
        for aborted_group in aborted_groups:
            state_queue.remove(aborted_group)
            for seq in aborted_group.get_seqs():
                if seq.is_finished():
                    continue
                seq.status = SequenceStatus.FINISHED_ABORTED
                self.free_seq(seq)