from __future__ import annotations
import functools
from .instruction import Instruction
from .gate import Gate
from .controlledgate import ControlledGate, _ctrl_state_to_int
@staticmethod
def _singleton_lookup_key(*_args, **_kwargs):
    """Given the arguments to the constructor, return a key tuple that identifies the singleton
        instance to retrieve, or ``None`` if the arguments imply that a mutable object must be
        created.

        For performance, as a special case, this method will not be called if the class constructor
        was given zero arguments (e.g. the construction ``XGate()`` will not call this method, but
        ``XGate(label=None)`` will), and the default singleton will immediately be returned.

        This static method can (and probably should) be overridden by subclasses.  The derived
        signature should match the class's ``__init__``; this method should then examine the
        arguments to determine whether it requires mutability, or what the cache key (if any) should
        be.

        The function should return either ``None`` or valid ``dict`` key (i.e. hashable and
        implements equality).  Returning ``None`` means that the created instance must be mutable.
        No further singleton-based processing will be done, and the class creation will proceed as
        if there was no singleton handling.  Otherwise, the returned key can be anything hashable
        and no special meaning is ascribed to it.  Whenever this method returns the same key, the
        same singleton instance will be returned.  We suggest that you use a tuple of the values of
        all arguments that can be set while maintaining the singleton nature.

        Only keys that match the default arguments or arguments given to ``additional_singletons``
        at class-creation time will actually return singletons; other values will return a standard
        mutable instance.

        .. note::

            The singleton machinery will handle an unhashable return from this function gracefully
            by returning a mutable instance.  Subclasses should ensure that their key is hashable in
            the happy path, but they do not need to manually verify that the user-supplied arguments
            are hashable.  For example, it's safe to implement this as::

                @staticmethod
                def _singleton_lookup_key(*args, **kwargs):
                    return None if kwargs else args

            even though a user might give some unhashable type as one of the ``args``.
        """
    return None