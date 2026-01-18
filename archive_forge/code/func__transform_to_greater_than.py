import numbers
from torch.distributions import constraints, transforms
@biject_to.register(constraints.greater_than)
@biject_to.register(constraints.greater_than_eq)
@transform_to.register(constraints.greater_than)
@transform_to.register(constraints.greater_than_eq)
def _transform_to_greater_than(constraint):
    return transforms.ComposeTransform([transforms.ExpTransform(), transforms.AffineTransform(constraint.lower_bound, 1)])