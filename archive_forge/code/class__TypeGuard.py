import abc
import collections
import collections.abc
import operator
import sys
import typing
class _TypeGuard(typing._FinalTypingBase, _root=True):
    """Special typing form used to annotate the return type of a user-defined
        type guard function.  ``TypeGuard`` only accepts a single type argument.
        At runtime, functions marked this way should return a boolean.

        ``TypeGuard`` aims to benefit *type narrowing* -- a technique used by static
        type checkers to determine a more precise type of an expression within a
        program's code flow.  Usually type narrowing is done by analyzing
        conditional code flow and applying the narrowing to a block of code.  The
        conditional expression here is sometimes referred to as a "type guard".

        Sometimes it would be convenient to use a user-defined boolean function
        as a type guard.  Such a function should use ``TypeGuard[...]`` as its
        return type to alert static type checkers to this intention.

        Using  ``-> TypeGuard`` tells the static type checker that for a given
        function:

        1. The return value is a boolean.
        2. If the return value is ``True``, the type of its argument
        is the type inside ``TypeGuard``.

        For example::

            def is_str(val: Union[str, float]):
                # "isinstance" type guard
                if isinstance(val, str):
                    # Type of ``val`` is narrowed to ``str``
                    ...
                else:
                    # Else, type of ``val`` is narrowed to ``float``.
                    ...

        Strict type narrowing is not enforced -- ``TypeB`` need not be a narrower
        form of ``TypeA`` (it can even be a wider form) and this may lead to
        type-unsafe results.  The main reason is to allow for things like
        narrowing ``List[object]`` to ``List[str]`` even though the latter is not
        a subtype of the former, since ``List`` is invariant.  The responsibility of
        writing type-safe type guards is left to the user.

        ``TypeGuard`` also works with type variables.  For more information, see
        PEP 647 (User-Defined Type Guards).
        """
    __slots__ = ('__type__',)

    def __init__(self, tp=None, **kwds):
        self.__type__ = tp

    def __getitem__(self, item):
        cls = type(self)
        if self.__type__ is None:
            return cls(typing._type_check(item, f'{cls.__name__[1:]} accepts only a single type.'), _root=True)
        raise TypeError(f'{cls.__name__[1:]} cannot be further subscripted')

    def _eval_type(self, globalns, localns):
        new_tp = typing._eval_type(self.__type__, globalns, localns)
        if new_tp == self.__type__:
            return self
        return type(self)(new_tp, _root=True)

    def __repr__(self):
        r = super().__repr__()
        if self.__type__ is not None:
            r += f'[{typing._type_repr(self.__type__)}]'
        return r

    def __hash__(self):
        return hash((type(self).__name__, self.__type__))

    def __eq__(self, other):
        if not isinstance(other, _TypeGuard):
            return NotImplemented
        if self.__type__ is not None:
            return self.__type__ == other.__type__
        return self is other