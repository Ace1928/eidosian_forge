import dataclasses
import io
import logging
import operator
from collections import ChainMap
from functools import reduce
from typing import List, Tuple, Dict, Any, Union, cast
import torch
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.planner import (
from torch.distributed.checkpoint.metadata import (
from torch.distributed.checkpoint.planner_helpers import (
from torch.distributed.checkpoint._nested_dict import (
from torch.distributed.checkpoint._sharded_tensor_utils import (
from torch.distributed.checkpoint._dedup_tensors import dedup_tensors
from torch.distributed.checkpoint.utils import find_state_dict_object
from torch.distributed.checkpoint._traverse import set_element
def _validate_global_plan(global_plan: List[SavePlan], metadata: Metadata) -> bool:
    all_good = True
    for key, value in metadata.state_dict_metadata.items():
        if isinstance(value, BytesStorageMetadata):
            continue
        if len(value.size) == 0:
            continue
        chunks_volume = 0
        for chunk_idx, chunk0 in enumerate(value.chunks):
            if not _check_box_bounds(value.size, chunk0):
                logger.warning('\n                        key:%s has out of bounds chunk:\n                        tensor-size:%s chunk: %s\n                    ', key, value.size, chunk0)
                all_good = False
            chunks_volume += reduce(operator.mul, chunk0.sizes, 1)
            for chunk1 in value.chunks[chunk_idx + 1:]:
                if _check_box_overlap(chunk0, chunk1):
                    logger.warning('key:%s has overlapping chunks: %s %s', key, chunk0, chunk1)
                    all_good = False
        tensor_volume = reduce(operator.mul, value.size, 1)
        if chunks_volume != tensor_volume:
            logger.warning('\n                    key:%s invalid fill tensor-volume:\n                    %s chunks-volume: %s\n                ', key, tensor_volume, chunks_volume)
            all_good = False
    return all_good