import abc
import os
import sys
import _collections_abc
from collections import deque
from functools import wraps
from types import MethodType, GenericAlias
class _GeneratorContextManager(_GeneratorContextManagerBase, AbstractContextManager, ContextDecorator):
    """Helper for @contextmanager decorator."""

    def __enter__(self):
        del self.args, self.kwds, self.func
        try:
            return next(self.gen)
        except StopIteration:
            raise RuntimeError("generator didn't yield") from None

    def __exit__(self, typ, value, traceback):
        if typ is None:
            try:
                next(self.gen)
            except StopIteration:
                return False
            else:
                try:
                    raise RuntimeError("generator didn't stop")
                finally:
                    self.gen.close()
        else:
            if value is None:
                value = typ()
            try:
                self.gen.throw(typ, value, traceback)
            except StopIteration as exc:
                return exc is not value
            except RuntimeError as exc:
                if exc is value:
                    exc.__traceback__ = traceback
                    return False
                if isinstance(value, StopIteration) and exc.__cause__ is value:
                    value.__traceback__ = traceback
                    return False
                raise
            except BaseException as exc:
                if exc is not value:
                    raise
                exc.__traceback__ = traceback
                return False
            try:
                raise RuntimeError("generator didn't stop after throw()")
            finally:
                self.gen.close()