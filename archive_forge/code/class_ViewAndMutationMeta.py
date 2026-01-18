import collections
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, NewType, Optional, Set, Union
import torch
import torch.utils._pytree as pytree
from torch._guards import Source
from torch._subclasses import FakeTensor
from torch._subclasses.fake_tensor import is_fake
from .. import config
from .utils import strict_zip
@dataclass(eq=False)
class ViewAndMutationMeta:
    input_info: List[InputAliasInfo]
    output_info: List[OutputAliasInfo]
    num_intermediate_bases: int
    keep_input_mutations: bool
    traced_tangents: List[Any]
    subclass_inp_meta: List[Union[int, SubclassCreationMeta]]
    subclass_fw_graph_out_meta: List[Union[int, SubclassCreationMeta]]
    subclass_tangent_meta: List[Union[int, SubclassCreationMeta]]
    is_train: bool = False
    requires_subclass_dispatch: bool = False
    num_symints_saved_for_bw: Optional[int] = None
    grad_enabled_mutation: Optional[bool] = None

    def __post_init__(self):
        if not self.requires_subclass_dispatch:
            mutated_inp_runtime_indices = [i for i, m in enumerate(self.input_info) if m.mutation_type == MutationType.MUTATED_OUT_GRAPH]
        else:
            mutated_inp_runtime_indices = [i for i, m in enumerate(self.input_info) if m.mutation_type in (MutationType.MUTATED_IN_GRAPH, MutationType.MUTATED_OUT_GRAPH)]
        mutated_graph_handled_indices = [i for i, m in enumerate(self.input_info) if m.mutation_type == MutationType.MUTATED_IN_GRAPH and (not self.requires_subclass_dispatch)]
        self.mutated_graph_handled_indices = mutated_graph_handled_indices
        self.num_mutated_graph_handled_indices = len(self.mutated_graph_handled_indices)
        aliased_out_indices = [i for i, m in enumerate(self.output_info) if m.output_type not in [OutputType.non_alias, OutputType.unsafe_view_alias, OutputType.custom_function_view]]
        unsafe_view_out_indices = [i for i, m in enumerate(self.output_info) if m.output_type is OutputType.unsafe_view_alias]
        self.mutated_inp_runtime_indices = mutated_inp_runtime_indices
        self.num_mutated_inp_runtime_indices = len(self.mutated_inp_runtime_indices)
        self.aliased_out_indices = aliased_out_indices
        self.unsafe_view_out_indices = unsafe_view_out_indices
        self.num_outputs = len(self.output_info)
        self.num_outputs_non_aliased = len([x for x in self.output_info if x.output_type in [OutputType.non_alias, OutputType.unsafe_view_alias, OutputType.custom_function_view]])
        self.num_outputs_aliased_to_inputs = len([x for x in self.output_info if x.output_type in [OutputType.alias_of_input, OutputType.is_input]])
        self.num_unsafe_view_outputs = len(self.unsafe_view_out_indices)
        self.num_outputs_aliased_to_intermediates = len([x for x in self.output_info if x.output_type in [OutputType.alias_of_intermediate, OutputType.alias_of_intermediate_save_as_output, OutputType.alias_of_intermediate_base_is_user_output]])
        self.num_outputs_aliased = self.num_outputs_aliased_to_inputs + self.num_outputs_aliased_to_intermediates
        self.dynamic_outputs = any((o.dynamic_dims for o in self.output_info))
        self.output_types = [torch.Tensor if isinstance(x, FakeTensor) else type(x) for x in self.traced_tangents]
        self.is_rng_op_functionalized = config.functionalize_rng_ops
        self.num_outputs_rng_offset = 1 if self.is_rng_op_functionalized else 0
        self.num_forward_returns = self.num_mutated_inp_runtime_indices + self.num_outputs + self.num_intermediate_bases
        self.num_forward = self.num_forward_returns + self.num_outputs_rng_offset

    @property
    def tensors_saved_for_backwards_slice(self):
        assert self.num_symints_saved_for_bw is not None
        if self.num_symints_saved_for_bw > 0:
            return slice(self.num_forward, -self.num_symints_saved_for_bw)
        else:
            return slice(self.num_forward, None)

    @property
    def symints_saved_for_backwards_slice(self):
        assert self.num_symints_saved_for_bw is not None
        if self.num_symints_saved_for_bw > 0:
            return slice(-self.num_symints_saved_for_bw, None)
        else:
            return slice(0, 0)

    def __eq__(self, other):
        if not isinstance(other, ViewAndMutationMeta):
            return NotImplemented
        return self.input_info == other.input_info and self.output_info == other.output_info and (self.num_intermediate_bases == other.num_intermediate_bases) and (self.keep_input_mutations == other.keep_input_mutations) and (self.is_rng_op_functionalized == other.is_rng_op_functionalized) and (self.num_outputs_rng_offset == other.num_outputs_rng_offset) and (len(self.traced_tangents) == len(other.traced_tangents)) and all((x.shape == y.shape and x.dtype == y.dtype for x, y in zip(self.traced_tangents, other.traced_tangents)))