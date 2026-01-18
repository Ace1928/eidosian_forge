import collections
import dataclasses
import itertools
import logging
import re
import typing
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import sympy
import torch
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from .codegen.common import index_prevent_reordering
from .utils import get_dtype_size, sympy_str, sympy_subs, sympy_symbol, VarRanges
from .virtualized import V
@dataclasses.dataclass
class ReadWrites:
    reads: Set[Dep]
    writes: Set[Dep]
    index_exprs: Set[IndexExprDep]
    range_vars: Optional[List[sympy.Expr]] = None
    var_ranges: Optional[VarRanges] = None
    op_counts: typing.Counter[str] = dataclasses.field(default_factory=collections.Counter)

    def rename(self, renames: typing.Dict[str, str]) -> 'ReadWrites':
        return ReadWrites({dep.rename(renames) for dep in self.reads}, {dep.rename(renames) for dep in self.writes}, self.index_exprs, self.range_vars, self.var_ranges, op_counts=self.op_counts)

    def with_read(self, dep: Dep) -> 'ReadWrites':
        assert isinstance(dep, (WeakDep, StarDep))
        return ReadWrites(set.union(self.reads, {dep}), self.writes, self.index_exprs, self.range_vars, self.var_ranges, op_counts=self.op_counts)

    def merge(self, other: 'ReadWrites'):
        reads = set.union(self.reads, other.reads)
        writes = set.union(self.writes, other.writes)
        index_exprs = set.union(self.index_exprs, other.index_exprs)
        op_counts = collections.Counter(self.op_counts)
        op_counts.update(other.op_counts)
        return ReadWrites(reads - writes, writes, index_exprs, op_counts=op_counts)

    @staticmethod
    def merge_list(read_writes: List['ReadWrites']):
        all_writes = set.union(*[rw.writes for rw in read_writes])
        all_reads = set.union(*[rw.reads for rw in read_writes]) - all_writes
        all_index_exprs = set.union(*[rw.index_exprs for rw in read_writes])
        op_counts: typing.Counter[Any] = collections.Counter()
        for rw in read_writes:
            op_counts.update(rw.op_counts)
        return ReadWrites(all_reads, all_writes, all_index_exprs, op_counts=op_counts)

    def remove_reads(self, rem_reads):
        return ReadWrites(self.reads - rem_reads, self.writes, self.index_exprs, self.range_vars, self.var_ranges, op_counts=self.op_counts)

    def reads_and_writes(self):
        return itertools.chain(self.reads, self.writes)