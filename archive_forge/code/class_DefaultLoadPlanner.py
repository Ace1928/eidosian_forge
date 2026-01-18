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
class DefaultLoadPlanner(LoadPlanner):
    """
    DefaultLoadPlanner that adds multiple features on top of LoadPlanner.

    In particular it adds the following:

    flatten_state_dict: Handle state_dict with nested dicts
    flatten_sharded_tensors: For FSDP in 2D parallel mode
    """
    original_state_dict: STATE_DICT_TYPE
    mappings: FLATTEN_MAPPING

    def __init__(self, flatten_state_dict: bool=True, flatten_sharded_tensors: bool=True) -> None:
        self.flatten_state_dict = flatten_state_dict
        self.flatten_sharded_tensors = flatten_sharded_tensors
        self.original_state_dict = {}
        self.mappings = {}

    def set_up_planner(self, state_dict: STATE_DICT_TYPE, metadata: Metadata, is_coordinator: bool) -> None:
        self.original_state_dict = state_dict
        if self.flatten_sharded_tensors:
            state_dict = _flatten_sharded_tensors(state_dict)
        if self.flatten_state_dict:
            state_dict, self.mappings = flatten_state_dict(state_dict)
        self.state_dict = state_dict
        self.metadata = metadata
        self.is_coordinator = is_coordinator

    def create_local_plan(self) -> LoadPlan:
        return create_default_local_load_plan(self.state_dict, self.metadata)

    def create_global_plan(self, global_plan: List[LoadPlan]) -> List[LoadPlan]:
        return create_default_global_load_plan(global_plan)

    def finish_plan(self, new_plan: LoadPlan) -> LoadPlan:
        return new_plan

    def load_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None:
        if self.flatten_state_dict:
            set_element(self.original_state_dict, self.mappings[read_item.dest_index.fqn], torch.load(value))
        else:
            self.state_dict[read_item.dest_index.fqn] = torch.load(value)

    def resolve_tensor(self, read_item: ReadItem):
        tensor = self.lookup_tensor(read_item.dest_index)
        return self.transform_tensor(read_item, tensor)

    def commit_tensor(self, read_item: ReadItem, tensor: torch.Tensor) -> None:
        pass

    def lookup_tensor(self, index: MetadataIndex) -> torch.Tensor:
        """Extension from the planner interface to make it easy to extend the default planner."""
        return find_state_dict_object(self.state_dict, index)

    def transform_tensor(self, read_item: ReadItem, tensor: torch.Tensor):
        """Extension from the planner interface to make it easy to extend the default planner."""
        return narrow_tensor_by_index(tensor, read_item.dest_offsets, read_item.lengths)