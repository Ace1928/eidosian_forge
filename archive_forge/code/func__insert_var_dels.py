from functools import cached_property
from numba.core import ir, analysis, transforms, ir_utils
def _insert_var_dels(self, extend_lifetimes=False):
    """
        Insert del statements for each variable.
        Returns a 2-tuple of (variable definition map, variable deletion map)
        which indicates variables defined and deleted in each block.

        The algorithm avoids relying on explicit knowledge on loops and
        distinguish between variables that are defined locally vs variables that
        come from incoming blocks.
        We start with simple usage (variable reference) and definition (variable
        creation) maps on each block. Propagate the liveness info to predecessor
        blocks until it stabilize, at which point we know which variables must
        exist before entering each block. Then, we compute the end of variable
        lives and insert del statements accordingly. Variables are deleted after
        the last use. Variable referenced by terminators (e.g. conditional
        branch and return) are deleted by the successors or the caller.
        """
    vlt = self.func_ir.variable_lifetime
    self._patch_var_dels(vlt.deadmaps.internal, vlt.deadmaps.escaping, extend_lifetimes=extend_lifetimes)