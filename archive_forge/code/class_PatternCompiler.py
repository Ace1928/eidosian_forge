import threading
import fasteners
from oslo_utils import excutils
from taskflow import flow
from taskflow import logging
from taskflow import task
from taskflow.types import graph as gr
from taskflow.types import tree as tr
from taskflow.utils import iter_utils
from taskflow.utils import misc
from taskflow.flow import (LINK_INVARIANT, LINK_RETRY)  # noqa
class PatternCompiler(object):
    """Compiles a flow pattern (or task) into a compilation unit.

    Let's dive into the basic idea for how this works:

    The compiler here is provided a 'root' object via its __init__ method,
    this object could be a task, or a flow (one of the supported patterns),
    the end-goal is to produce a :py:class:`.Compilation` object as the result
    with the needed components. If this is not possible a
    :py:class:`~.taskflow.exceptions.CompilationFailure` will be raised.
    In the case where a **unknown** type is being requested to compile
    a ``TypeError`` will be raised and when a duplicate object (one that
    has **already** been compiled) is encountered a ``ValueError`` is raised.

    The complexity of this comes into play when the 'root' is a flow that
    contains itself other nested flows (and so-on); to compile this object and
    its contained objects into a graph that *preserves* the constraints the
    pattern mandates we have to go through a recursive algorithm that creates
    subgraphs for each nesting level, and then on the way back up through
    the recursion (now with a decomposed mapping from contained patterns or
    atoms to there corresponding subgraph) we have to then connect the
    subgraphs (and the atom(s) there-in) that were decomposed for a pattern
    correctly into a new graph and then ensure the pattern mandated
    constraints are retained. Finally we then return to the
    caller (and they will do the same thing up until the root node, which by
    that point one graph is created with all contained atoms in the
    pattern/nested patterns mandated ordering).

    Also maintained in the :py:class:`.Compilation` object is a hierarchy of
    the nesting of items (which is also built up during the above mentioned
    recusion, via a much simpler algorithm); this is typically used later to
    determine the prior atoms of a given atom when looking up values that can
    be provided to that atom for execution (see the scopes.py file for how this
    works). Note that although you *could* think that the graph itself could be
    used for this, which in some ways it can (for limited usage) the hierarchy
    retains the nested structure (which is useful for scoping analysis/lookup)
    to be able to provide back a iterator that gives back the scopes visible
    at each level (the graph does not have this information once flattened).

    Let's take an example:

    Given the pattern ``f(a(b, c), d)`` where ``f`` is a
    :py:class:`~taskflow.patterns.linear_flow.Flow` with items ``a(b, c)``
    where ``a`` is a :py:class:`~taskflow.patterns.linear_flow.Flow` composed
    of tasks ``(b, c)`` and task ``d``.

    The algorithm that will be performed (mirroring the above described logic)
    will go through the following steps (the tree hierarchy building is left
    out as that is more obvious)::

        Compiling f
          - Decomposing flow f with no parent (must be the root)
          - Compiling a
              - Decomposing flow a with parent f
              - Compiling b
                  - Decomposing task b with parent a
                  - Decomposed b into:
                    Name: b
                    Nodes: 1
                      - b
                    Edges: 0
              - Compiling c
                  - Decomposing task c with parent a
                  - Decomposed c into:
                    Name: c
                    Nodes: 1
                      - c
                    Edges: 0
              - Relinking decomposed b -> decomposed c
              - Decomposed a into:
                Name: a
                Nodes: 2
                  - b
                  - c
                Edges: 1
                  b -> c ({'invariant': True})
          - Compiling d
              - Decomposing task d with parent f
              - Decomposed d into:
                Name: d
                Nodes: 1
                  - d
                Edges: 0
          - Relinking decomposed a -> decomposed d
          - Decomposed f into:
            Name: f
            Nodes: 3
              - c
              - b
              - d
            Edges: 2
              c -> d ({'invariant': True})
              b -> c ({'invariant': True})
    """

    def __init__(self, root, freeze=True):
        self._root = root
        self._history = set()
        self._freeze = freeze
        self._lock = threading.Lock()
        self._compilation = None
        self._matchers = [(flow.Flow, FlowCompiler(self._compile)), (task.Task, TaskCompiler())]
        self._level = 0

    def _compile(self, item, parent=None):
        """Compiles a item (pattern, task) into a graph + tree node."""
        item_compiler = misc.match_type(item, self._matchers)
        if item_compiler is not None:
            self._pre_item_compile(item)
            graph, node = item_compiler.compile(item, parent=parent)
            self._post_item_compile(item, graph, node)
            return (graph, node)
        else:
            raise TypeError("Unknown object '%s' (%s) requested to compile" % (item, type(item)))

    def _pre_item_compile(self, item):
        """Called before a item is compiled; any pre-compilation actions."""
        if item in self._history:
            raise ValueError("Already compiled item '%s' (%s), duplicate and/or recursive compiling is not supported" % (item, type(item)))
        self._history.add(item)
        if LOG.isEnabledFor(logging.TRACE):
            LOG.trace("%sCompiling '%s'", '  ' * self._level, item)
        self._level += 1

    def _post_item_compile(self, item, graph, node):
        """Called after a item is compiled; doing post-compilation actions."""
        self._level -= 1
        if LOG.isEnabledFor(logging.TRACE):
            prefix = '  ' * self._level
            LOG.trace("%sDecomposed '%s' into:", prefix, item)
            prefix = '  ' * (self._level + 1)
            LOG.trace('%sGraph:', prefix)
            for line in graph.pformat().splitlines():
                LOG.trace('%s  %s', prefix, line)
            LOG.trace('%sHierarchy:', prefix)
            for line in node.pformat().splitlines():
                LOG.trace('%s  %s', prefix, line)

    def _pre_compile(self):
        """Called before the compilation of the root starts."""
        self._history.clear()
        self._level = 0

    def _post_compile(self, graph, node):
        """Called after the compilation of the root finishes successfully."""
        self._history.clear()
        self._level = 0

    @fasteners.locked
    def compile(self):
        """Compiles the contained item into a compiled equivalent."""
        if self._compilation is None:
            self._pre_compile()
            try:
                graph, node = self._compile(self._root, parent=None)
            except Exception:
                with excutils.save_and_reraise_exception():
                    self._history.clear()
            else:
                self._post_compile(graph, node)
                if self._freeze:
                    graph.freeze()
                    node.freeze()
                self._compilation = Compilation(graph, node)
        return self._compilation