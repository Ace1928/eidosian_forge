import abc
from collections.abc import Mapping
from typing import TypeVar, Generic
from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.basic_block import (
class RegionTransformer(abc.ABC, Generic[Tdata]):
    """A mutating pass over a SCFG.

    This class is similar to ``RegionVisitor`` but only a "forward" direction
    is supported.
    """

    @abc.abstractmethod
    def visit_block(self, parent: SCFG, block: BasicBlock, data: Tdata) -> Tdata:
        pass

    @abc.abstractmethod
    def visit_loop(self, parent: SCFG, region: RegionBlock, data: Tdata) -> Tdata:
        pass

    @abc.abstractmethod
    def visit_switch(self, parent: SCFG, region: RegionBlock, data: Tdata) -> Tdata:
        pass

    def visit_linear(self, parent: SCFG, region: RegionBlock, data: Tdata) -> Tdata:
        return self.visit_graph(region.subregion, data)

    def visit_graph(self, scfg: SCFG, data: Tdata) -> Tdata:
        toposorted = toposort_graph(scfg.graph)
        label: str
        for lvl in toposorted:
            for label in lvl:
                data = self.visit(scfg, scfg[label], data)
        return data

    def visit(self, parent: SCFG, block: BasicBlock, data: Tdata) -> Tdata:
        if isinstance(block, RegionBlock):
            if block.kind == 'loop':
                fn = self.visit_loop
            elif block.kind == 'switch':
                fn = self.visit_switch
            elif block.kind in {'head', 'tail', 'branch'}:
                fn = self.visit_linear
            else:
                raise NotImplementedError('unreachable', block.name, block.kind)
            data = fn(parent, block, data)
        else:
            data = self.visit_block(parent, block, data)
        return data