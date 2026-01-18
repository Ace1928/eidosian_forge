from __future__ import absolute_import, print_function
import sys
import inspect
from . import TypeSlots
from . import Builtin
from . import Nodes
from . import ExprNodes
from . import Errors
from . import DebugFlags
from . import Future
import cython
def _raise_compiler_error(self, child, e):
    trace = ['']
    for parent, attribute, index in self.access_path:
        node = getattr(parent, attribute)
        if index is None:
            index = ''
        else:
            node = node[index]
            index = u'[%d]' % index
        trace.append(u'%s.%s%s = %s' % (parent.__class__.__name__, attribute, index, self.dump_node(node)))
    stacktrace, called_nodes = self._find_node_path(sys.exc_info()[2])
    last_node = child
    for node, method_name, pos in called_nodes:
        last_node = node
        trace.append(u"File '%s', line %d, in %s: %s" % (pos[0], pos[1], method_name, self.dump_node(node)))
    raise Errors.CompilerCrash(getattr(last_node, 'pos', None), self.__class__.__name__, u'\n'.join(trace), e, stacktrace)