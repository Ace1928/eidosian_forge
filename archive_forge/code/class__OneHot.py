import torch
class _OneHot(Constraint):
    """
    Constrain to one-hot vectors.
    """
    is_discrete = True
    event_dim = 1

    def check(self, value):
        is_boolean = (value == 0) | (value == 1)
        is_normalized = value.sum(-1).eq(1)
        return is_boolean.all(-1) & is_normalized