from typing import (
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe
from enum import IntEnum
class _ShardingIterDataPipe(IterDataPipe):

    def apply_sharding(self, num_of_instances: int, instance_id: int, sharding_group: SHARDING_PRIORITIES):
        raise NotImplementedError