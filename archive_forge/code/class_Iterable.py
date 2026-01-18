from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
class Iterable(object):
    """Interface for distributed objects that admit iteration/reduction."""

    def __iter__(self):
        pass

    def reduce(self, initial_state, reduce_func):
        """Reduces this iterable object to a single element.

    The transformation calls `reduce_func` successively on each element.
    The `initial_state` argument is used for the initial state and the final
    state is returned as the result.

    Args:
      initial_state: An element representing the initial state of the
        reduction.
      reduce_func: A function that maps `(old_state, input_element)` to
        `new_state`. The structure of `new_state` must match the structure of
        `old_state`. For the first element, `old_state` is `initial_state`.

    Returns:
      The final state of the transformation.
    """