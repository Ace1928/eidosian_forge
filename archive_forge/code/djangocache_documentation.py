from functools import wraps
from django.core.cache.backends.base import BaseCache
from .core import ENOVAL, args_to_key, full_name
from .fanout import FanoutCache
Make key for cache given function arguments.