import enum
from typing import Dict, List, Optional, Set, Tuple
from vllm.block import BlockTable, PhysicalTokenBlock
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device
def _free_block_table(self, block_table: BlockTable) -> None:
    for block in set(block_table):
        if block.device == Device.GPU:
            self.gpu_allocator.free(block)
        else:
            self.cpu_allocator.free(block)