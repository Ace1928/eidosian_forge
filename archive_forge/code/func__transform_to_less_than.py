import numbers
from torch.distributions import constraints, transforms
@biject_to.register(constraints.less_than)
@transform_to.register(constraints.less_than)
def _transform_to_less_than(constraint):
    return transforms.ComposeTransform([transforms.ExpTransform(), transforms.AffineTransform(constraint.upper_bound, -1)])