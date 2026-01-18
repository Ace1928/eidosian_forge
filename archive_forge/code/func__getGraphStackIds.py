import threading
from tensorboard import errors
def _getGraphStackIds(self, graph_id):
    """Retrieve the IDs of all outer graphs of a graph.

        Args:
          graph_id: Id of the graph being queried with respect to its outer
            graphs context.

        Returns:
          A list of graph_ids, ordered from outermost to innermost, including
            the input `graph_id` argument as the last item.
        """
    graph_ids = [graph_id]
    graph = self._reader.graph_by_id(graph_id)
    while graph.outer_graph_id:
        graph_ids.insert(0, graph.outer_graph_id)
        graph = self._reader.graph_by_id(graph.outer_graph_id)
    return graph_ids