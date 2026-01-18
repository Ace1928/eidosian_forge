from ray.data._internal.logical.interfaces import LogicalOperator
class NAry(LogicalOperator):
    """Base class for n-ary operators, which take multiple input operators."""

    def __init__(self, *input_ops: LogicalOperator):
        """
        Args:
            input_ops: The input operators.
        """
        super().__init__(self.__class__.__name__, list(input_ops))