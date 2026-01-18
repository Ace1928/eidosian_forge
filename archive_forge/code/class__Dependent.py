import torch
class _Dependent(Constraint):
    """
    Placeholder for variables whose support depends on other variables.
    These variables obey no simple coordinate-wise constraints.

    Args:
        is_discrete (bool): Optional value of ``.is_discrete`` in case this
            can be computed statically. If not provided, access to the
            ``.is_discrete`` attribute will raise a NotImplementedError.
        event_dim (int): Optional value of ``.event_dim`` in case this
            can be computed statically. If not provided, access to the
            ``.event_dim`` attribute will raise a NotImplementedError.
    """

    def __init__(self, *, is_discrete=NotImplemented, event_dim=NotImplemented):
        self._is_discrete = is_discrete
        self._event_dim = event_dim
        super().__init__()

    @property
    def is_discrete(self):
        if self._is_discrete is NotImplemented:
            raise NotImplementedError('.is_discrete cannot be determined statically')
        return self._is_discrete

    @property
    def event_dim(self):
        if self._event_dim is NotImplemented:
            raise NotImplementedError('.event_dim cannot be determined statically')
        return self._event_dim

    def __call__(self, *, is_discrete=NotImplemented, event_dim=NotImplemented):
        """
        Support for syntax to customize static attributes::

            constraints.dependent(is_discrete=True, event_dim=1)
        """
        if is_discrete is NotImplemented:
            is_discrete = self._is_discrete
        if event_dim is NotImplemented:
            event_dim = self._event_dim
        return _Dependent(is_discrete=is_discrete, event_dim=event_dim)

    def check(self, x):
        raise ValueError('Cannot determine validity of dependent constraint')