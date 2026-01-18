import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
def _process_basic_statement(self, node):
    self.generic_visit(node)
    self.builder.add_ordinary_node(node)