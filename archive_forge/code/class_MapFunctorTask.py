import abc
import copy
import functools
from oslo_utils import reflection
from taskflow import atom
from taskflow import logging
from taskflow.types import notifier
from taskflow.utils import misc
class MapFunctorTask(Task):
    """General purpose Task to map a function to a list.

    This Task mimics the behavior of Python's built-in ``map`` function. The
    Task takes a functor (lambda or otherwise) and a list. The list is
    specified using the ``requires`` argument of the Task. When executed, this
    task calls ``map`` with the functor and list as arguments. The resulting
    list from the call to ``map`` is then returned after execution.

    Each value of the returned list can be bound to individual names using
    the ``provides`` argument, following taskflow standard behavior. Order is
    preserved in the returned list.
    """

    def __init__(self, functor, requires, name=None, provides=None, auto_extract=True, rebind=None, inject=None):
        if not callable(functor):
            raise ValueError('Function to use for map must be callable')
        f_args = reflection.get_callable_args(functor)
        if len(f_args) != 1:
            raise ValueError('%s arguments were provided. Map functor must take exactly 1 argument.' % len(f_args))
        if not misc.is_iterable(requires):
            raise TypeError('%s type was provided for requires. Requires must be an iterable.' % type(requires))
        if name is None:
            name = reflection.get_callable_name(functor)
        super(MapFunctorTask, self).__init__(name=name, provides=provides, inject=inject, requires=requires, rebind=rebind, auto_extract=auto_extract)
        self._functor = functor

    def execute(self, *args, **kwargs):
        l = [kwargs[r] for r in self.requires]
        return list(map(self._functor, l))