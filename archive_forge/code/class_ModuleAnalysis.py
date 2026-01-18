import gast as ast
import os
import re
from time import time
class ModuleAnalysis(Analysis):
    """An analysis that operates on a whole module."""

    def run(self, node):
        if not isinstance(node, ast.Module):
            if self.ctx.module is None:
                raise ValueError('{} called in an uninitialized context'.format(type(self).__name__))
            node = self.ctx.module
        return super(ModuleAnalysis, self).run(node)