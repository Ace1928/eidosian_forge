from __future__ import annotations
import typing as ty
from nibabel.deprecated import deprecate_with_version
class ResetMixin:
    """A Mixin class to add a .reset() method to users of OneTimeProperty.

    By default, auto attributes once computed, become static.  If they happen
    to depend on other parts of an object and those parts change, their values
    may now be invalid.

    This class offers a .reset() method that users can call *explicitly* when
    they know the state of their objects may have changed and they want to
    ensure that *all* their special attributes should be invalidated.  Once
    reset() is called, all their auto attributes are reset to their
    OneTimeProperty descriptors, and their accessor functions will be triggered
    again.

    .. warning::

       If a class has a set of attributes that are OneTimeProperty, but that
       can be initialized from any one of them, do NOT use this mixin!  For
       instance, UniformTimeSeries can be initialized with only sampling_rate
       and t0, sampling_interval and time are auto-computed.  But if you were
       to reset() a UniformTimeSeries, it would lose all 4, and there would be
       then no way to break the circular dependency chains.

       If this becomes a problem in practice (for our analyzer objects it
       isn't, as they don't have the above pattern), we can extend reset() to
       check for a _no_reset set of names in the instance which are meant to be
       kept protected.  But for now this is NOT done, so caveat emptor.

    Examples
    --------

    >>> class A(ResetMixin):
    ...     def __init__(self,x=1.0):
    ...         self.x = x
    ...
    ...     @auto_attr
    ...     def y(self):
    ...         print('*** y computation executed ***')
    ...         return self.x / 2.0
    ...

    >>> a = A(10)

    About to access y twice, the second time no computation is done:
    >>> a.y
    *** y computation executed ***
    5.0
    >>> a.y
    5.0

    Changing x
    >>> a.x = 20

    a.y doesn't change to 10, since it is a static attribute:
    >>> a.y
    5.0

    We now reset a, and this will then force all auto attributes to recompute
    the next time we access them:
    >>> a.reset()

    About to access y twice again after reset():
    >>> a.y
    *** y computation executed ***
    10.0
    >>> a.y
    10.0
    """

    def reset(self) -> None:
        """Reset all OneTimeProperty attributes that may have fired already."""
        for mname, mval in self.__class__.__dict__.items():
            if mname in self.__dict__ and isinstance(mval, OneTimeProperty):
                delattr(self, mname)