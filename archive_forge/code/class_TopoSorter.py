from . import errors
from . import graph as _mod_graph
from . import revision as _mod_revision
class TopoSorter:

    def __init__(self, graph):
        """Topological sorting of a graph.

        :param graph: sequence of pairs of node_name->parent_names_list.
                      i.e. [('C', ['B']), ('B', ['A']), ('A', [])]
                      For this input the output from the sort or
                      iter_topo_order routines will be:
                      'A', 'B', 'C'

        node identifiers can be any hashable object, and are typically strings.

        If you have a graph like [('a', ['b']), ('a', ['c'])] this will only use
        one of the two values for 'a'.

        The graph is sorted lazily: until you iterate or sort the input is
        not processed other than to create an internal representation.

        iteration or sorting may raise GraphCycleError if a cycle is present
        in the graph.
        """
        self._graph = dict(graph)

    def sorted(self):
        """Sort the graph and return as a list.

        After calling this the sorter is empty and you must create a new one.
        """
        return list(self.iter_topo_order())

    def iter_topo_order(self):
        """Yield the nodes of the graph in a topological order.

        After finishing iteration the sorter is empty and you cannot continue
        iteration.
        """
        graph = self._graph
        visitable = set(graph)
        pending_node_stack = []
        pending_parents_stack = []
        completed_node_names = set()
        while graph:
            node_name, parents = graph.popitem()
            pending_node_stack.append(node_name)
            pending_parents_stack.append(list(parents))
            while pending_node_stack:
                parents_to_visit = pending_parents_stack[-1]
                if not parents_to_visit:
                    popped_node = pending_node_stack.pop()
                    pending_parents_stack.pop()
                    completed_node_names.add(popped_node)
                    yield popped_node
                else:
                    next_node_name = parents_to_visit.pop()
                    if next_node_name in completed_node_names:
                        continue
                    if next_node_name not in visitable:
                        continue
                    try:
                        parents = graph.pop(next_node_name)
                    except KeyError:
                        raise errors.GraphCycleError(pending_node_stack)
                    pending_node_stack.append(next_node_name)
                    pending_parents_stack.append(list(parents))