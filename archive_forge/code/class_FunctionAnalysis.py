import gast as ast
import os
import re
from time import time
class FunctionAnalysis(Analysis):
    """An analysis that operates on a function."""

    def run(self, node):
        if isinstance(node, ast.Module):
            self.ctx.module = node
            last = None
            for stmt in node.body:
                if isinstance(stmt, ast.FunctionDef):
                    last = self.gather(type(self), stmt)
            return self.result if last is None else last
        elif not isinstance(node, ast.FunctionDef):
            if self.ctx.function is None:
                raise ValueError('{} called in an uninitialized context'.format(type(self).__name__))
            node = self.ctx.function
        return super(FunctionAnalysis, self).run(node)