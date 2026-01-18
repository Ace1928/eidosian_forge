import traceback
from contextvars import copy_context
from . import BaseLongCallbackManager
from ..._callback_context import context_value
from ..._utils import AttributeDict
from ...exceptions import PreventUpdate

        Long callback manager that runs callback logic in a subprocess and stores
        results on disk using diskcache

        :param cache:
            A diskcache.Cache or diskcache.FanoutCache instance. See the diskcache
            documentation for information on configuration options. If not provided,
            a diskcache.Cache instance will be created with default values.
        :param cache_by:
            A list of zero-argument functions.  When provided, caching is enabled and
            the return values of these functions are combined with the callback
            function's input arguments and source code to generate cache keys.
        :param expire:
            If provided, a cache entry will be removed when it has not been accessed
            for ``expire`` seconds.  If not provided, the lifetime of cache entries
            is determined by the default behavior of the ``cache`` instance.
        