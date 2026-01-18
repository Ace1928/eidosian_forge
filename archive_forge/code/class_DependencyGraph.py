from collections import Counter
from textwrap import dedent
from kombu.utils.encoding import bytes_to_str, safe_str
class DependencyGraph:
    """A directed acyclic graph of objects and their dependencies.

    Supports a robust topological sort
    to detect the order in which they must be handled.

    Takes an optional iterator of ``(obj, dependencies)``
    tuples to build the graph from.

    Warning:
        Does not support cycle detection.
    """

    def __init__(self, it=None, formatter=None):
        self.formatter = formatter or GraphFormatter()
        self.adjacent = {}
        if it is not None:
            self.update(it)

    def add_arc(self, obj):
        """Add an object to the graph."""
        self.adjacent.setdefault(obj, [])

    def add_edge(self, A, B):
        """Add an edge from object ``A`` to object ``B``.

        I.e. ``A`` depends on ``B``.
        """
        self[A].append(B)

    def connect(self, graph):
        """Add nodes from another graph."""
        self.adjacent.update(graph.adjacent)

    def topsort(self):
        """Sort the graph topologically.

        Returns:
            List: of objects in the order in which they must be handled.
        """
        graph = DependencyGraph()
        components = self._tarjan72()
        NC = {node: component for component in components for node in component}
        for component in components:
            graph.add_arc(component)
        for node in self:
            node_c = NC[node]
            for successor in self[node]:
                successor_c = NC[successor]
                if node_c != successor_c:
                    graph.add_edge(node_c, successor_c)
        return [t[0] for t in graph._khan62()]

    def valency_of(self, obj):
        """Return the valency (degree) of a vertex in the graph."""
        try:
            l = [len(self[obj])]
        except KeyError:
            return 0
        for node in self[obj]:
            l.append(self.valency_of(node))
        return sum(l)

    def update(self, it):
        """Update graph with data from a list of ``(obj, deps)`` tuples."""
        tups = list(it)
        for obj, _ in tups:
            self.add_arc(obj)
        for obj, deps in tups:
            for dep in deps:
                self.add_edge(obj, dep)

    def edges(self):
        """Return generator that yields for all edges in the graph."""
        return (obj for obj, adj in self.items() if adj)

    def _khan62(self):
        """Perform Khan's simple topological sort algorithm from '62.

        See https://en.wikipedia.org/wiki/Topological_sorting
        """
        count = Counter()
        result = []
        for node in self:
            for successor in self[node]:
                count[successor] += 1
        ready = [node for node in self if not count[node]]
        while ready:
            node = ready.pop()
            result.append(node)
            for successor in self[node]:
                count[successor] -= 1
                if count[successor] == 0:
                    ready.append(successor)
        result.reverse()
        return result

    def _tarjan72(self):
        """Perform Tarjan's algorithm to find strongly connected components.

        See Also:
            :wikipedia:`Tarjan%27s_strongly_connected_components_algorithm`
        """
        result, stack, low = ([], [], {})

        def visit(node):
            if node in low:
                return
            num = len(low)
            low[node] = num
            stack_pos = len(stack)
            stack.append(node)
            for successor in self[node]:
                visit(successor)
                low[node] = min(low[node], low[successor])
            if num == low[node]:
                component = tuple(stack[stack_pos:])
                stack[stack_pos:] = []
                result.append(component)
                for item in component:
                    low[item] = len(self)
        for node in self:
            visit(node)
        return result

    def to_dot(self, fh, formatter=None):
        """Convert the graph to DOT format.

        Arguments:
            fh (IO): A file, or a file-like object to write the graph to.
            formatter (celery.utils.graph.GraphFormatter): Custom graph
                formatter to use.
        """
        seen = set()
        draw = formatter or self.formatter

        def P(s):
            print(bytes_to_str(s), file=fh)

        def if_not_seen(fun, obj):
            if draw.label(obj) not in seen:
                P(fun(obj))
                seen.add(draw.label(obj))
        P(draw.head())
        for obj, adjacent in self.items():
            if not adjacent:
                if_not_seen(draw.terminal_node, obj)
            for req in adjacent:
                if_not_seen(draw.node, obj)
                P(draw.edge(obj, req))
        P(draw.tail())

    def format(self, obj):
        return self.formatter(obj) if self.formatter else obj

    def __iter__(self):
        return iter(self.adjacent)

    def __getitem__(self, node):
        return self.adjacent[node]

    def __len__(self):
        return len(self.adjacent)

    def __contains__(self, obj):
        return obj in self.adjacent

    def _iterate_items(self):
        return self.adjacent.items()
    items = iteritems = _iterate_items

    def __repr__(self):
        return '\n'.join((self.repr_node(N) for N in self))

    def repr_node(self, obj, level=1, fmt='{0}({1})'):
        output = [fmt.format(obj, self.valency_of(obj))]
        if obj in self:
            for other in self[obj]:
                d = fmt.format(other, self.valency_of(other))
                output.append('     ' * level + d)
                output.extend(self.repr_node(other, level + 1).split('\n')[1:])
        return '\n'.join(output)