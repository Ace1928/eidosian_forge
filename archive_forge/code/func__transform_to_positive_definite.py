import numbers
from torch.distributions import constraints, transforms
@transform_to.register(constraints.positive_definite)
@transform_to.register(constraints.positive_semidefinite)
def _transform_to_positive_definite(constraint):
    return transforms.PositiveDefiniteTransform()