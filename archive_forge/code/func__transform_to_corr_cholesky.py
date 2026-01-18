import numbers
from torch.distributions import constraints, transforms
@biject_to.register(constraints.corr_cholesky)
@transform_to.register(constraints.corr_cholesky)
def _transform_to_corr_cholesky(constraint):
    return transforms.CorrCholeskyTransform()