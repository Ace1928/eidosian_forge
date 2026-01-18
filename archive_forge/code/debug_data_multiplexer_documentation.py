import threading
from tensorboard import errors
Retrieve the IDs of all outer graphs of a graph.

        Args:
          graph_id: Id of the graph being queried with respect to its outer
            graphs context.

        Returns:
          A list of graph_ids, ordered from outermost to innermost, including
            the input `graph_id` argument as the last item.
        