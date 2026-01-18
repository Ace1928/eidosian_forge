import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
def _process_exit_statement(self, node, exits_nodes_of_type, may_exit_via_except=False):
    self.generic_visit(node)
    try_node, guards = self._get_enclosing_finally_scopes(exits_nodes_of_type)
    assert try_node is not None, '{} that is not enclosed by any of {}'.format(node, exits_nodes_of_type)
    node = self.builder.add_exit_node(node, try_node, guards)
    if may_exit_via_except:
        except_guards = self._get_enclosing_except_scopes(exits_nodes_of_type)
        self.builder.connect_raise_node(node, except_guards)