from __future__ import annotations
import asyncio
import inspect
import math
import operator
from collections.abc import Iterable, Iterator
from functools import partial
from types import FunctionType, MethodType
from typing import Any, Callable, Optional
from .depends import depends
from .display import _display_accessors, _reactive_display_objs
from .parameterized import (
from .parameters import Boolean, Event
from ._utils import _to_async_gen, iscoroutinefunction, full_groupby
class reactive_ops:
    """
    Namespace for reactive operators.

    Implements operators that cannot be implemented using regular
    Python syntax.
    """

    def __init__(self, reactive):
        self._reactive = reactive

    def _as_rx(self):
        return self._reactive if isinstance(self._reactive, rx) else self()

    def __call__(self):
        rxi = self._reactive
        return rxi if isinstance(rx, rx) else rx(rxi)

    def and_(self, other):
        """
        Replacement for the ``and`` statement.
        """
        return self._as_rx()._apply_operator(lambda obj, other: obj and other, other)

    def bool(self):
        """
        __bool__ cannot be implemented so it is provided as a method.
        """
        return self._as_rx()._apply_operator(bool)

    def buffer(self, n):
        """
        Collects the last n items that were emmitted.
        """
        items = []

        def collect(new, n):
            items.append(new)
            while len(items) > n:
                items.pop(0)
            return items
        return self._as_rx()._apply_operator(collect, n)

    def in_(self, other):
        """
        Replacement for the ``in`` statement.
        """
        return self._as_rx()._apply_operator(operator.contains, other, reverse=True)

    def is_(self, other):
        """
        Replacement for the ``is`` statement.
        """
        return self._as_rx()._apply_operator(operator.is_, other)

    def is_not(self, other):
        """
        Replacement for the ``is not`` statement.
        """
        return self._as_rx()._apply_operator(operator.is_not, other)

    def len(self):
        """
        __len__ cannot be implemented so it is provided as a method.
        """
        return self._as_rx()._apply_operator(len)

    def map(self, func, /, *args, **kwargs):
        """
        Apply a function to each item.

        Arguments
        ---------
        func: function
          Function to apply.
        args: iterable, optional
          Positional arguments to pass to `func`.
        kwargs: mapping, optional
          A dictionary of keywords to pass to `func`.
        """
        if inspect.isasyncgenfunction(func) or inspect.isgeneratorfunction(func):
            raise TypeError('Cannot map a generator function. Only regular function or coroutine functions are permitted.')
        if inspect.iscoroutinefunction(func):

            async def apply(vs, *args, **kwargs):
                return list(await asyncio.gather(*(func(v, *args, **kwargs) for v in vs)))
        else:

            def apply(vs, *args, **kwargs):
                return [func(v, *args, **kwargs) for v in vs]
        return self._as_rx()._apply_operator(apply, *args, **kwargs)

    def not_(self):
        """
        __bool__ cannot be implemented so not has to be provided as a method.
        """
        return self._as_rx()._apply_operator(operator.not_)

    def or_(self, other):
        """
        Replacement for the ``or`` statement.
        """
        return self._as_rx()._apply_operator(lambda obj, other: obj or other, other)

    def pipe(self, func, /, *args, **kwargs):
        """
        Apply chainable functions.

        Arguments
        ---------
        func: function
          Function to apply.
        args: iterable, optional
          Positional arguments to pass to `func`.
        kwargs: mapping, optional
          A dictionary of keywords to pass to `func`.
        """
        return self._as_rx()._apply_operator(func, *args, **kwargs)

    def resolve(self, nested=True, recursive=False):
        """
        Resolves references held by the expression.

        As an example if the expression returns a list of parameters
        this operation will return a list of the parameter values.

        Arguments
        ---------
        nested: bool
          Whether to resolve references contained within nested objects,
          i.e. tuples, lists, sets and dictionaries.
        recursive: bool
          Whether to recursively resolve references, i.e. if a reference
          itself returns a reference we recurse into it until no more
          references can be resolved.
        """
        resolver_type = NestedResolver if nested else Resolver
        resolver = resolver_type(object=self._reactive, recursive=recursive)
        return resolver.param.value.rx()

    def updating(self):
        """
        Returns a new expression that is True while the expression is updating.
        """
        wrapper = Wrapper(object=False)
        self._watch(lambda e: wrapper.param.update(object=True), precedence=-999)
        self._watch(lambda e: wrapper.param.update(object=False), precedence=999)
        return wrapper.param.object.rx()

    def when(self, *dependencies, initial=Undefined):
        """
        Returns a reactive expression that emits the contents of this
        expression only when the dependencies change. If initial value
        is provided and the dependencies are all param.Event types the
        expression will not be evaluated until the first event is
        triggered.

        Arguments
        ---------
        dependencies: param.Parameter | rx
          A dependency that will trigger an update in the output.
        initial: object
          Object that will stand in for the actual value until the
          first time a param.Event in the dependencies is triggered.
        """
        deps = [p for d in dependencies for p in resolve_ref(d)]
        is_event = all((isinstance(dep, Event) for dep in deps))

        def eval(*_, evaluated=[]):
            if is_event and initial is not Undefined and (not evaluated):
                evaluated.append(True)
                return initial
            else:
                return self.value
        return bind(eval, *deps).rx()

    def where(self, x, y):
        """
        Returns either x or y depending on the current state of the
        expression, i.e. replaces a ternary if statement.

        Arguments
        ---------
        x: object
          The value to return if the expression evaluates to True.
        y: object
          The value to return if the expression evaluates to False.
        """
        xrefs = resolve_ref(x)
        yrefs = resolve_ref(y)
        if isinstance(self._reactive, rx):
            params = self._reactive._params
        else:
            params = resolve_ref(self._reactive)
        trigger = Trigger(parameters=params)
        if xrefs:

            def trigger_x(*args):
                if self.value:
                    trigger.param.trigger('value')
            bind(trigger_x, *xrefs, watch=True)
        if yrefs:

            def trigger_y(*args):
                if not self.value:
                    trigger.param.trigger('value')
            bind(trigger_y, *yrefs, watch=True)

        def ternary(condition, _):
            return resolve_value(x) if condition else resolve_value(y)
        return bind(ternary, self._reactive, trigger.param.value)

    @property
    def value(self):
        """
        Returns the current state of the reactive expression by
        evaluating the pipeline.
        """
        if isinstance(self._reactive, rx):
            return self._reactive._resolve()
        elif isinstance(self._reactive, Parameter):
            return getattr(self._reactive.owner, self._reactive.name)
        else:
            return self._reactive()

    @value.setter
    def value(self, new):
        """
        Allows overriding the original input to the pipeline.
        """
        if isinstance(self._reactive, Parameter):
            raise AttributeError('`Parameter.rx.value = value` is not supported. Cannot override parameter value.')
        elif not isinstance(self._reactive, rx):
            raise AttributeError('`bind(...).rx.value = value` is not supported. Cannot override the output of a function.')
        elif self._reactive._root is not self._reactive:
            raise AttributeError('The value of a derived expression cannot be set. Ensure you set the value on the root node wrapping a concrete value, e.g.:\n\n    a = rx(1)\n    b = a + 1\n    a.rx.value = 2\n\n is valid but you may not set `b.rx.value = 2`.')
        if self._reactive._wrapper is None:
            raise AttributeError('Setting the value of a reactive expression is only supported if it wraps a concrete value. A reactive expression wrapping a Parameter or another dynamic reference cannot be updated.')
        self._reactive._wrapper.object = resolve_value(new)

    def watch(self, fn=None, onlychanged=True, queued=False, precedence=0):
        """
        Adds a callable that observes the output of the pipeline.
        If no callable is provided this simply causes the expression
        to be eagerly evaluated.
        """
        if precedence < 0:
            raise ValueError('User-defined watch callbacks must declare a positive precedence. Negative precedences are reserved for internal Watchers.')
        self._watch(fn, onlychanged=onlychanged, queued=queued, precedence=precedence)

    def _watch(self, fn=None, onlychanged=True, queued=False, precedence=0):

        def cb(value):
            from .parameterized import async_executor
            if iscoroutinefunction(fn):
                async_executor(partial(fn, value))
            elif fn is not None:
                fn(value)
        bind(cb, self._reactive, watch=True)