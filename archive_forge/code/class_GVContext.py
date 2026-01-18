from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
class GVContext(object):
    """Graphviz subgraph object."""

    def __init__(self):
        self.blockids = {}
        self.nextid = 0
        self.children = []
        self.sources = {}

    def add(self, child):
        self.children.append(child)

    def nodeid(self, block):
        if block not in self.blockids:
            self.blockids[block] = 'block%d' % self.nextid
            self.nextid += 1
        return self.blockids[block]

    def extract_sources(self, block):
        if not block.positions:
            return ''
        start = min(block.positions)
        stop = max(block.positions)
        srcdescr = start[0]
        if srcdescr not in self.sources:
            self.sources[srcdescr] = list(srcdescr.get_lines())
        lines = self.sources[srcdescr]
        return '\\n'.join([l.strip() for l in lines[start[1] - 1:stop[1]]])

    def render(self, fp, name, annotate_defs=False):
        """Render graphviz dot graph"""
        fp.write('digraph %s {\n' % name)
        fp.write(' node [shape=box];\n')
        for child in self.children:
            child.render(fp, self, annotate_defs)
        fp.write('}\n')

    def escape(self, text):
        return text.replace('"', '\\"').replace('\n', '\\n')