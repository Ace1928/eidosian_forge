import torch
class _IndependentConstraint(Constraint):
    """
    Wraps a constraint by aggregating over ``reinterpreted_batch_ndims``-many
    dims in :meth:`check`, so that an event is valid only if all its
    independent entries are valid.
    """

    def __init__(self, base_constraint, reinterpreted_batch_ndims):
        assert isinstance(base_constraint, Constraint)
        assert isinstance(reinterpreted_batch_ndims, int)
        assert reinterpreted_batch_ndims >= 0
        self.base_constraint = base_constraint
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        super().__init__()

    @property
    def is_discrete(self):
        return self.base_constraint.is_discrete

    @property
    def event_dim(self):
        return self.base_constraint.event_dim + self.reinterpreted_batch_ndims

    def check(self, value):
        result = self.base_constraint.check(value)
        if result.dim() < self.reinterpreted_batch_ndims:
            expected = self.base_constraint.event_dim + self.reinterpreted_batch_ndims
            raise ValueError(f'Expected value.dim() >= {expected} but got {value.dim()}')
        result = result.reshape(result.shape[:result.dim() - self.reinterpreted_batch_ndims] + (-1,))
        result = result.all(-1)
        return result

    def __repr__(self):
        return f'{self.__class__.__name__[1:]}({repr(self.base_constraint)}, {self.reinterpreted_batch_ndims})'