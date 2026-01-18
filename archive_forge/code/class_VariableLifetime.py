from functools import cached_property
from numba.core import ir, analysis, transforms, ir_utils
class VariableLifetime(object):
    """
    For lazily building information of variable lifetime
    """

    def __init__(self, blocks):
        self._blocks = blocks

    @cached_property
    def cfg(self):
        return analysis.compute_cfg_from_blocks(self._blocks)

    @cached_property
    def usedefs(self):
        return analysis.compute_use_defs(self._blocks)

    @cached_property
    def livemap(self):
        return analysis.compute_live_map(self.cfg, self._blocks, self.usedefs.usemap, self.usedefs.defmap)

    @cached_property
    def deadmaps(self):
        return analysis.compute_dead_maps(self.cfg, self._blocks, self.livemap, self.usedefs.defmap)