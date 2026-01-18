import torch
class _Cat(Constraint):
    """
    Constraint functor that applies a sequence of constraints
    `cseq` at the submatrices at dimension `dim`,
    each of size `lengths[dim]`, in a way compatible with :func:`torch.cat`.
    """

    def __init__(self, cseq, dim=0, lengths=None):
        assert all((isinstance(c, Constraint) for c in cseq))
        self.cseq = list(cseq)
        if lengths is None:
            lengths = [1] * len(self.cseq)
        self.lengths = list(lengths)
        assert len(self.lengths) == len(self.cseq)
        self.dim = dim
        super().__init__()

    @property
    def is_discrete(self):
        return any((c.is_discrete for c in self.cseq))

    @property
    def event_dim(self):
        return max((c.event_dim for c in self.cseq))

    def check(self, value):
        assert -value.dim() <= self.dim < value.dim()
        checks = []
        start = 0
        for constr, length in zip(self.cseq, self.lengths):
            v = value.narrow(self.dim, start, length)
            checks.append(constr.check(v))
            start = start + length
        return torch.cat(checks, self.dim)