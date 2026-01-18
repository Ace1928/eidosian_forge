import abc
import functools
import re
import tokenize
from tokenize import untokenize, TokenError
from io import StringIO
from IPython.core.splitinput import LineInfo
from IPython.utils import tokenutil
class InputTransformer(metaclass=abc.ABCMeta):
    """Abstract base class for line-based input transformers."""

    @abc.abstractmethod
    def push(self, line):
        """Send a line of input to the transformer, returning the transformed
        input or None if the transformer is waiting for more input.

        Must be overridden by subclasses.

        Implementations may raise ``SyntaxError`` if the input is invalid. No
        other exceptions may be raised.
        """
        pass

    @abc.abstractmethod
    def reset(self):
        """Return, transformed any lines that the transformer has accumulated,
        and reset its internal state.

        Must be overridden by subclasses.
        """
        pass

    @classmethod
    def wrap(cls, func):
        """Can be used by subclasses as a decorator, to return a factory that
        will allow instantiation with the decorated object.
        """

        @functools.wraps(func)
        def transformer_factory(**kwargs):
            return cls(func, **kwargs)
        return transformer_factory