import numbers
from torch.distributions import constraints, transforms
@transform_to.register(constraints.independent)
def _transform_to_independent(constraint):
    base_transform = transform_to(constraint.base_constraint)
    return transforms.IndependentTransform(base_transform, constraint.reinterpreted_batch_ndims)