import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _determine_transform(self, cfg, block, label, scope, init_equiv_set):
    """Determine the transformation for each instruction in the block
        """
    equiv_set = None
    preds = cfg.predecessors(label)
    if label in self.pruned_predecessors:
        pruned = self.pruned_predecessors[label]
    else:
        pruned = []
    if config.DEBUG_ARRAY_OPT >= 2:
        print('preds:', preds)
    for p, q in preds:
        if config.DEBUG_ARRAY_OPT >= 2:
            print('p, q:', p, q)
        if p in pruned:
            continue
        if p in self.equiv_sets:
            from_set = self.equiv_sets[p].clone()
            if config.DEBUG_ARRAY_OPT >= 2:
                print('p in equiv_sets', from_set)
            if (p, label) in self.prepends:
                instrs = self.prepends[p, label]
                for inst in instrs:
                    redefined = set()
                    self._analyze_inst(label, scope, from_set, inst, redefined)
                    self.remove_redefineds(redefined)
            if equiv_set is None:
                equiv_set = from_set
            else:
                equiv_set = equiv_set.intersect(from_set)
                redefined = set()
                equiv_set.union_defs(from_set.defs, redefined)
                self.remove_redefineds(redefined)
    if equiv_set is None:
        equiv_set = init_equiv_set
    self.equiv_sets[label] = equiv_set
    pending_transforms = []
    for inst in block.body:
        redefined = set()
        pre, post = self._analyze_inst(label, scope, equiv_set, inst, redefined)
        if len(redefined) > 0:
            self.remove_redefineds(redefined)
        pending_transforms.append((inst, pre, post))
    return pending_transforms