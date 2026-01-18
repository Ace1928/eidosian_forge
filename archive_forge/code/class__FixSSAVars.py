import logging
import operator
import warnings
from functools import reduce
from copy import copy
from pprint import pformat
from collections import defaultdict
from numba import config
from numba.core import ir, ir_utils, errors
from numba.core.analysis import compute_cfg_from_blocks
class _FixSSAVars(_BaseHandler):
    """Replace variable uses in IR nodes to the correct reaching variable
    and introduce Phi nodes if necessary. This class contains the core of
    the SSA reconstruction algorithm.

    See Ch 5 of the Inria SSA book for reference. The method names used here
    are similar to the names used in the pseudocode in the book.
    """

    def __init__(self, cache_list_vars):
        self._cache_list_vars = cache_list_vars

    def on_assign(self, states, assign):
        rhs = assign.value
        if isinstance(rhs, ir.Inst):
            newdef = self._fix_var(states, assign, self._cache_list_vars.get(assign.value))
            if newdef is not None and newdef.target is not ir.UNDEFINED:
                if states['varname'] != newdef.target.name:
                    replmap = {states['varname']: newdef.target}
                    rhs = copy(rhs)
                    ir_utils.replace_vars_inner(rhs, replmap)
                    return ir.Assign(target=assign.target, value=rhs, loc=assign.loc)
        elif isinstance(rhs, ir.Var):
            newdef = self._fix_var(states, assign, [rhs])
            if newdef is not None and newdef.target is not ir.UNDEFINED:
                if states['varname'] != newdef.target.name:
                    return ir.Assign(target=assign.target, value=newdef.target, loc=assign.loc)
        return assign

    def on_other(self, states, stmt):
        newdef = self._fix_var(states, stmt, self._cache_list_vars.get(stmt))
        if newdef is not None and newdef.target is not ir.UNDEFINED:
            if states['varname'] != newdef.target.name:
                replmap = {states['varname']: newdef.target}
                stmt = copy(stmt)
                ir_utils.replace_vars_stmt(stmt, replmap)
        return stmt

    def _fix_var(self, states, stmt, used_vars):
        """Fix all variable uses in ``used_vars``.
        """
        varnames = [k.name for k in used_vars]
        phivar = states['varname']
        if phivar in varnames:
            return self._find_def(states, stmt)

    def _find_def(self, states, stmt):
        """Find definition of ``stmt`` for the statement ``stmt``
        """
        _logger.debug('find_def var=%r stmt=%s', states['varname'], stmt)
        selected_def = None
        label = states['label']
        local_defs = states['defmap'][label]
        local_phis = states['phimap'][label]
        block = states['block']
        cur_pos = self._stmt_index(stmt, block)
        for defstmt in reversed(local_defs):
            def_pos = self._stmt_index(defstmt, block, stop=cur_pos)
            if def_pos < cur_pos:
                selected_def = defstmt
                break
            elif defstmt in local_phis:
                selected_def = local_phis[-1]
                break
        if selected_def is None:
            selected_def = self._find_def_from_top(states, label, loc=stmt.loc)
        return selected_def

    def _find_def_from_top(self, states, label, loc):
        """Find definition reaching block of ``label``.

        This method would look at all dominance frontiers.
        Insert phi node if necessary.
        """
        _logger.debug('find_def_from_top label %r', label)
        cfg = states['cfg']
        defmap = states['defmap']
        phimap = states['phimap']
        phi_locations = states['phi_locations']
        if label in phi_locations:
            scope = states['scope']
            loc = states['block'].loc
            freshvar = scope.redefine(states['varname'], loc=loc)
            phinode = ir.Assign(target=freshvar, value=ir.Expr.phi(loc=loc), loc=loc)
            _logger.debug('insert phi node %s at %s', phinode, label)
            defmap[label].insert(0, phinode)
            phimap[label].append(phinode)
            for pred, _ in cfg.predecessors(label):
                incoming_def = self._find_def_from_bottom(states, pred, loc=loc)
                _logger.debug('incoming_def %s', incoming_def)
                phinode.value.incoming_values.append(incoming_def.target)
                phinode.value.incoming_blocks.append(pred)
            return phinode
        else:
            idom = cfg.immediate_dominators()[label]
            if idom == label:
                _warn_about_uninitialized_variable(states['varname'], loc)
                return UndefinedVariable
            _logger.debug('idom %s from label %s', idom, label)
            return self._find_def_from_bottom(states, idom, loc=loc)

    def _find_def_from_bottom(self, states, label, loc):
        """Find definition from within the block at ``label``.
        """
        _logger.debug('find_def_from_bottom label %r', label)
        defmap = states['defmap']
        defs = defmap[label]
        if defs:
            lastdef = defs[-1]
            return lastdef
        else:
            return self._find_def_from_top(states, label, loc=loc)

    def _stmt_index(self, defstmt, block, stop=-1):
        """Find the positional index of the statement at ``block``.

        Assumptions:
        - no two statements can point to the same object.
        """
        for i in range(len(block.body))[:stop]:
            if block.body[i] is defstmt:
                return i
        return len(block.body)