import abc
import copy
import functools
from oslo_utils import reflection
from taskflow import atom
from taskflow import logging
from taskflow.types import notifier
from taskflow.utils import misc
class ReduceFunctorTask(Task):
    """General purpose Task to reduce a list by applying a function.

    This Task mimics the behavior of Python's built-in ``reduce`` function. The
    Task takes a functor (lambda or otherwise) and a list. The list is
    specified using the ``requires`` argument of the Task. When executed, this
    task calls ``reduce`` with the functor and list as arguments. The resulting
    value from the call to ``reduce`` is then returned after execution.
    """

    def __init__(self, functor, requires, name=None, provides=None, auto_extract=True, rebind=None, inject=None):
        if not callable(functor):
            raise ValueError('Function to use for reduce must be callable')
        f_args = reflection.get_callable_args(functor)
        if len(f_args) != 2:
            raise ValueError('%s arguments were provided. Reduce functor must take exactly 2 arguments.' % len(f_args))
        if not misc.is_iterable(requires):
            raise TypeError('%s type was provided for requires. Requires must be an iterable.' % type(requires))
        if len(requires) < 2:
            raise ValueError('%s elements were provided. Requires must have at least 2 elements.' % len(requires))
        if name is None:
            name = reflection.get_callable_name(functor)
        super(ReduceFunctorTask, self).__init__(name=name, provides=provides, inject=inject, requires=requires, rebind=rebind, auto_extract=auto_extract)
        self._functor = functor

    def execute(self, *args, **kwargs):
        l = [kwargs[r] for r in self.requires]
        return functools.reduce(self._functor, l)