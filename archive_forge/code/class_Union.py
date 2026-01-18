from ray.data._internal.logical.interfaces import LogicalOperator
class Union(NAry):
    """Logical operator for union."""

    def __init__(self, *input_ops: LogicalOperator):
        super().__init__(*input_ops)