from functools import cached_property
from numba.core import ir, analysis, transforms, ir_utils
def _compute_generator_info(self):
    """
        Compute the generator's state variables as the union of live variables
        at all yield points.
        """
    self._insert_var_dels()
    self._populate_generator_info()
    gi = self.func_ir.generator_info
    for yp in gi.get_yield_points():
        live_vars = set(self.func_ir.get_block_entry_vars(yp.block))
        weak_live_vars = set()
        stmts = iter(yp.block.body)
        for stmt in stmts:
            if isinstance(stmt, ir.Assign):
                if stmt.value is yp.inst:
                    break
                live_vars.add(stmt.target.name)
            elif isinstance(stmt, ir.Del):
                live_vars.remove(stmt.value)
        else:
            assert 0, "couldn't find yield point"
        for stmt in stmts:
            if isinstance(stmt, ir.Del):
                name = stmt.value
                if name in live_vars:
                    live_vars.remove(name)
                    weak_live_vars.add(name)
            else:
                break
        yp.live_vars = live_vars
        yp.weak_live_vars = weak_live_vars
    st = set()
    for yp in gi.get_yield_points():
        st |= yp.live_vars
        st |= yp.weak_live_vars
    gi.state_vars = sorted(st)
    self.remove_dels()