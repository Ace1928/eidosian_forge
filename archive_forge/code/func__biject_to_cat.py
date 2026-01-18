import numbers
from torch.distributions import constraints, transforms
@biject_to.register(constraints.cat)
def _biject_to_cat(constraint):
    return transforms.CatTransform([biject_to(c) for c in constraint.cseq], constraint.dim, constraint.lengths)