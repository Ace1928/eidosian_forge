import re
import sys
from typing import Any, Callable, Mapping, Optional, Tuple, Type, cast
from weakref import ref
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
class DeferredCallable(object):

    def __init__(self, func: Callable[..., Any]):
        self.func = func
        self.args: Tuple[Any, ...] = tuple()
        self.kwargs: Mapping[str, Any] = {}

    def __call__(self):
        self.func(*self.args, **self.kwargs)

    def with_args(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return self