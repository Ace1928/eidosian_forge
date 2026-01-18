from cupy._core import _fusion_variable
from cupy._core import _fusion_op
def _normalize_ashapes(ops, variables, shape_constraints):

    def normalize(shape):
        return tuple([shape_constraints.evaluate(d) for d in shape])
    for var in variables:
        var.ashape = normalize(var.ashape)
    for op in ops:
        if isinstance(op, _fusion_op._ElementwiseTraceOp):
            op.ashape = normalize(op.ashape)