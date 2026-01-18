import functools
import inspect
import warnings
import sys
from typing import Any, Callable, TypeVar, cast
def _wrap_generator(ctx_factory, func):
    """
    Wrap each generator invocation with the context manager factory.

    The input should be a function that returns a context manager,
    not a context manager itself, to handle one-shot context managers.
    """

    @functools.wraps(func)
    def generator_context(*args, **kwargs):
        gen = func(*args, **kwargs)
        try:
            with ctx_factory():
                response = gen.send(None)
            while True:
                try:
                    request = (yield response)
                except GeneratorExit:
                    with ctx_factory():
                        gen.close()
                    raise
                except BaseException:
                    with ctx_factory():
                        response = gen.throw(*sys.exc_info())
                else:
                    with ctx_factory():
                        response = gen.send(request)
        except StopIteration as e:
            return e.value
    return generator_context