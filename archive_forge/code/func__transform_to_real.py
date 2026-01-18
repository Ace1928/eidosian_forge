import numbers
from torch.distributions import constraints, transforms
@biject_to.register(constraints.real)
@transform_to.register(constraints.real)
def _transform_to_real(constraint):
    return transforms.identity_transform