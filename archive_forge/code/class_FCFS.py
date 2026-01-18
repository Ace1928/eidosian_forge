from collections import deque
from typing import Deque
from vllm.sequence import SequenceGroup
class FCFS(Policy):

    def get_priority(self, now: float, seq_group: SequenceGroup) -> float:
        return now - seq_group.metrics.arrival_time