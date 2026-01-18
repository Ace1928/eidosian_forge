import gast as ast
import os
import re
from time import time
class Analysis(ContextManager, ast.NodeVisitor):
    """
    A pass that does not change its content but gathers informations about it.
    """

    def __init__(self, *dependencies):
        """`dependencies' holds the type of all analysis required by this
            analysis. `self.result' must be set prior to calling this
            constructor."""
        assert hasattr(self, 'result'), 'An analysis must have a result attribute when initialized'
        self.update = False
        ContextManager.__init__(self, *dependencies)

    def run(self, node):
        key = (node, type(self))
        if key in self.passmanager._cache:
            self.result = self.passmanager._cache[key]
        else:
            super(Analysis, self).run(node)
            self.passmanager._cache[key] = self.result
        return self.result

    def display(self, data):
        print(data)

    def apply(self, node):
        self.display(self.run(node))
        return (False, node)