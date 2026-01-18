import torch
class _DependentProperty(property, _Dependent):
    """
    Decorator that extends @property to act like a `Dependent` constraint when
    called on a class and act like a property when called on an object.

    Example::

        class Uniform(Distribution):
            def __init__(self, low, high):
                self.low = low
                self.high = high
            @constraints.dependent_property(is_discrete=False, event_dim=0)
            def support(self):
                return constraints.interval(self.low, self.high)

    Args:
        fn (Callable): The function to be decorated.
        is_discrete (bool): Optional value of ``.is_discrete`` in case this
            can be computed statically. If not provided, access to the
            ``.is_discrete`` attribute will raise a NotImplementedError.
        event_dim (int): Optional value of ``.event_dim`` in case this
            can be computed statically. If not provided, access to the
            ``.event_dim`` attribute will raise a NotImplementedError.
    """

    def __init__(self, fn=None, *, is_discrete=NotImplemented, event_dim=NotImplemented):
        super().__init__(fn)
        self._is_discrete = is_discrete
        self._event_dim = event_dim

    def __call__(self, fn):
        """
        Support for syntax to customize static attributes::

            @constraints.dependent_property(is_discrete=True, event_dim=1)
            def support(self):
                ...
        """
        return _DependentProperty(fn, is_discrete=self._is_discrete, event_dim=self._event_dim)