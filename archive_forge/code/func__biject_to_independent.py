import numbers
from torch.distributions import constraints, transforms
@biject_to.register(constraints.independent)
def _biject_to_independent(constraint):
    base_transform = biject_to(constraint.base_constraint)
    return transforms.IndependentTransform(base_transform, constraint.reinterpreted_batch_ndims)