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
class TreeVisitor(object):
    """
    Base class for writing visitors for a Cython tree, contains utilities for
    recursing such trees using visitors. Each node is
    expected to have a child_attrs iterable containing the names of attributes
    containing child nodes or lists of child nodes. Lists are not considered
    part of the tree structure (i.e. contained nodes are considered direct
    children of the parent node).

    visit_children visits each of the children of a given node (see the visit_children
    documentation). When recursing the tree using visit_children, an attribute
    access_path is maintained which gives information about the current location
    in the tree as a stack of tuples: (parent_node, attrname, index), representing
    the node, attribute and optional list index that was taken in each step in the path to
    the current node.

    Example:

    >>> class SampleNode(object):
    ...     child_attrs = ["head", "body"]
    ...     def __init__(self, value, head=None, body=None):
    ...         self.value = value
    ...         self.head = head
    ...         self.body = body
    ...     def __repr__(self): return "SampleNode(%s)" % self.value
    ...
    >>> tree = SampleNode(0, SampleNode(1), [SampleNode(2), SampleNode(3)])
    >>> class MyVisitor(TreeVisitor):
    ...     def visit_SampleNode(self, node):
    ...         print("in %s %s" % (node.value, self.access_path))
    ...         self.visitchildren(node)
    ...         print("out %s" % node.value)
    ...
    >>> MyVisitor().visit(tree)
    in 0 []
    in 1 [(SampleNode(0), 'head', None)]
    out 1
    in 2 [(SampleNode(0), 'body', 0)]
    out 2
    in 3 [(SampleNode(0), 'body', 1)]
    out 3
    out 0
    """

    def __init__(self):
        super(TreeVisitor, self).__init__()
        self.dispatch_table = {}
        self.access_path = []

    def dump_node(self, node):
        ignored = list(node.child_attrs or []) + ['child_attrs', 'pos', 'gil_message', 'cpp_message', 'subexprs']
        values = []
        pos = getattr(node, 'pos', None)
        if pos:
            source = pos[0]
            if source:
                import os.path
                source = os.path.basename(source.get_description())
            values.append(u'%s:%s:%s' % (source, pos[1], pos[2]))
        attribute_names = dir(node)
        for attr in attribute_names:
            if attr in ignored:
                continue
            if attr.startswith('_') or attr.endswith('_'):
                continue
            try:
                value = getattr(node, attr)
            except AttributeError:
                continue
            if value is None or value == 0:
                continue
            elif isinstance(value, list):
                value = u'[...]/%d' % len(value)
            elif not isinstance(value, _PRINTABLE):
                continue
            else:
                value = repr(value)
            values.append(u'%s = %s' % (attr, value))
        return u'%s(%s)' % (node.__class__.__name__, u',\n    '.join(values))

    def _find_node_path(self, stacktrace):
        import os.path
        last_traceback = stacktrace
        nodes = []
        while hasattr(stacktrace, 'tb_frame'):
            frame = stacktrace.tb_frame
            node = frame.f_locals.get('self')
            if isinstance(node, Nodes.Node):
                code = frame.f_code
                method_name = code.co_name
                pos = (os.path.basename(code.co_filename), frame.f_lineno)
                nodes.append((node, method_name, pos))
                last_traceback = stacktrace
            stacktrace = stacktrace.tb_next
        return (last_traceback, nodes)

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

    @cython.final
    def find_handler(self, obj):
        cls = type(obj)
        mro = inspect.getmro(cls)
        for mro_cls in mro:
            handler_method = getattr(self, 'visit_' + mro_cls.__name__, None)
            if handler_method is not None:
                return handler_method
        print(type(self), cls)
        if self.access_path:
            print(self.access_path)
            print(self.access_path[-1][0].pos)
            print(self.access_path[-1][0].__dict__)
        raise RuntimeError('Visitor %r does not accept object: %s' % (self, obj))

    def visit(self, obj):
        return self._visit(obj)

    @cython.final
    def _visit(self, obj):
        try:
            try:
                handler_method = self.dispatch_table[type(obj)]
            except KeyError:
                handler_method = self.find_handler(obj)
                self.dispatch_table[type(obj)] = handler_method
            return handler_method(obj)
        except Errors.CompileError:
            raise
        except Errors.AbortError:
            raise
        except Exception as e:
            if DebugFlags.debug_no_exception_intercept:
                raise
            self._raise_compiler_error(obj, e)

    @cython.final
    def _visitchild(self, child, parent, attrname, idx):
        self.access_path.append((parent, attrname, idx))
        result = self._visit(child)
        self.access_path.pop()
        return result

    def visitchildren(self, parent, attrs=None, exclude=None):
        return self._visitchildren(parent, attrs, exclude)

    @cython.final
    @cython.locals(idx=cython.Py_ssize_t)
    def _visitchildren(self, parent, attrs, exclude):
        """
        Visits the children of the given parent. If parent is None, returns
        immediately (returning None).

        The return value is a dictionary giving the results for each
        child (mapping the attribute name to either the return value
        or a list of return values (in the case of multiple children
        in an attribute)).
        """
        if parent is None:
            return None
        result = {}
        for attr in parent.child_attrs:
            if attrs is not None and attr not in attrs:
                continue
            if exclude is not None and attr in exclude:
                continue
            child = getattr(parent, attr)
            if child is not None:
                if type(child) is list:
                    childretval = [self._visitchild(x, parent, attr, idx) for idx, x in enumerate(child)]
                else:
                    childretval = self._visitchild(child, parent, attr, None)
                    assert not isinstance(childretval, list), 'Cannot insert list here: %s in %r' % (attr, parent)
                result[attr] = childretval
        return result