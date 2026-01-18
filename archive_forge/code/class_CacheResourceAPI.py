from __future__ import annotations
import math
import threading
import types
from datetime import timedelta
from typing import Any, Callable, Final, TypeVar, cast, overload
from cachetools import TTLCache
from typing_extensions import TypeAlias
import streamlit as st
from streamlit.deprecation_util import show_deprecation_warning
from streamlit.logger import get_logger
from streamlit.runtime.caching import cache_utils
from streamlit.runtime.caching.cache_errors import CacheKeyNotFoundError
from streamlit.runtime.caching.cache_type import CacheType
from streamlit.runtime.caching.cache_utils import (
from streamlit.runtime.caching.cached_message_replay import (
from streamlit.runtime.caching.hashing import HashFuncsDict
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from streamlit.runtime.stats import CacheStat, CacheStatsProvider, group_stats
from streamlit.time_util import time_to_seconds
class CacheResourceAPI:
    """Implements the public st.cache_resource API: the @st.cache_resource decorator,
    and st.cache_resource.clear().
    """

    def __init__(self, decorator_metric_name: str, deprecation_warning: str | None=None):
        """Create a CacheResourceAPI instance.

        Parameters
        ----------
        decorator_metric_name
            The metric name to record for decorator usage. `@st.experimental_singleton` is
            deprecated, but we're still supporting it and tracking its usage separately
            from `@st.cache_resource`.

        deprecation_warning
            An optional deprecation warning to show when the API is accessed.
        """
        self._decorator = gather_metrics(decorator_metric_name, self._decorator)
        self._deprecation_warning = deprecation_warning
    F = TypeVar('F', bound=Callable[..., Any])

    @overload
    def __call__(self, func: F) -> F:
        ...

    @overload
    def __call__(self, *, ttl: float | timedelta | str | None=None, max_entries: int | None=None, show_spinner: bool | str=True, validate: ValidateFunc | None=None, experimental_allow_widgets: bool=False, hash_funcs: HashFuncsDict | None=None) -> Callable[[F], F]:
        ...

    def __call__(self, func: F | None=None, *, ttl: float | timedelta | str | None=None, max_entries: int | None=None, show_spinner: bool | str=True, validate: ValidateFunc | None=None, experimental_allow_widgets: bool=False, hash_funcs: HashFuncsDict | None=None):
        return self._decorator(func, ttl=ttl, max_entries=max_entries, show_spinner=show_spinner, validate=validate, experimental_allow_widgets=experimental_allow_widgets, hash_funcs=hash_funcs)

    def _decorator(self, func: F | None, *, ttl: float | timedelta | str | None, max_entries: int | None, show_spinner: bool | str, validate: ValidateFunc | None, experimental_allow_widgets: bool, hash_funcs: HashFuncsDict | None=None):
        """Decorator to cache functions that return global resources (e.g. database connections, ML models).

        Cached objects are shared across all users, sessions, and reruns. They
        must be thread-safe because they can be accessed from multiple threads
        concurrently. If thread safety is an issue, consider using ``st.session_state``
        to store resources per session instead.

        You can clear a function's cache with ``func.clear()`` or clear the entire
        cache with ``st.cache_resource.clear()``.

        To cache data, use ``st.cache_data`` instead. Learn more about caching at
        https://docs.streamlit.io/library/advanced-features/caching.

        Parameters
        ----------
        func : callable
            The function that creates the cached resource. Streamlit hashes the
            function's source code.

        ttl : float, timedelta, str, or None
            The maximum time to keep an entry in the cache. Can be one of:

            * ``None`` if cache entries should never expire (default).
            * A number specifying the time in seconds.
            * A string specifying the time in a format supported by `Pandas's
              Timedelta constructor <https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html>`_,
              e.g. ``"1d"``, ``"1.5 days"``, or ``"1h23s"``.
            * A ``timedelta`` object from `Python's built-in datetime library
              <https://docs.python.org/3/library/datetime.html#timedelta-objects>`_,
              e.g. ``timedelta(days=1)``.

        max_entries : int or None
            The maximum number of entries to keep in the cache, or None
            for an unbounded cache. When a new entry is added to a full cache,
            the oldest cached entry will be removed. Defaults to None.

        show_spinner : bool or str
            Enable the spinner. Default is True to show a spinner when there is
            a "cache miss" and the cached resource is being created. If string,
            value of show_spinner param will be used for spinner text.

        validate : callable or None
            An optional validation function for cached data. ``validate`` is called
            each time the cached value is accessed. It receives the cached value as
            its only parameter and it must return a boolean. If ``validate`` returns
            False, the current cached value is discarded, and the decorated function
            is called to compute a new value. This is useful e.g. to check the
            health of database connections.

        experimental_allow_widgets : bool
            Allow widgets to be used in the cached function. Defaults to False.
            Support for widgets in cached functions is currently experimental.
            Setting this parameter to True may lead to excessive memory use since the
            widget value is treated as an additional input parameter to the cache.
            We may remove support for this option at any time without notice.

        hash_funcs : dict or None
            Mapping of types or fully qualified names to hash functions.
            This is used to override the behavior of the hasher inside Streamlit's
            caching mechanism: when the hasher encounters an object, it will first
            check to see if its type matches a key in this dict and, if so, will use
            the provided function to generate a hash for it. See below for an example
            of how this can be used.

        Example
        -------
        >>> import streamlit as st
        >>>
        >>> @st.cache_resource
        ... def get_database_session(url):
        ...     # Create a database session object that points to the URL.
        ...     return session
        ...
        >>> s1 = get_database_session(SESSION_URL_1)
        >>> # Actually executes the function, since this is the first time it was
        >>> # encountered.
        >>>
        >>> s2 = get_database_session(SESSION_URL_1)
        >>> # Does not execute the function. Instead, returns its previously computed
        >>> # value. This means that now the connection object in s1 is the same as in s2.
        >>>
        >>> s3 = get_database_session(SESSION_URL_2)
        >>> # This is a different URL, so the function executes.

        By default, all parameters to a cache_resource function must be hashable.
        Any parameter whose name begins with ``_`` will not be hashed. You can use
        this as an "escape hatch" for parameters that are not hashable:

        >>> import streamlit as st
        >>>
        >>> @st.cache_resource
        ... def get_database_session(_sessionmaker, url):
        ...     # Create a database connection object that points to the URL.
        ...     return connection
        ...
        >>> s1 = get_database_session(create_sessionmaker(), DATA_URL_1)
        >>> # Actually executes the function, since this is the first time it was
        >>> # encountered.
        >>>
        >>> s2 = get_database_session(create_sessionmaker(), DATA_URL_1)
        >>> # Does not execute the function. Instead, returns its previously computed
        >>> # value - even though the _sessionmaker parameter was different
        >>> # in both calls.

        A cache_resource function's cache can be procedurally cleared:

        >>> import streamlit as st
        >>>
        >>> @st.cache_resource
        ... def get_database_session(_sessionmaker, url):
        ...     # Create a database connection object that points to the URL.
        ...     return connection
        ...
        >>> get_database_session.clear()
        >>> # Clear all cached entries for this function.

        To override the default hashing behavior, pass a custom hash function.
        You can do that by mapping a type (e.g. ``Person``) to a hash
        function (``str``) like this:

        >>> import streamlit as st
        >>> from pydantic import BaseModel
        >>>
        >>> class Person(BaseModel):
        ...     name: str
        >>>
        >>> @st.cache_resource(hash_funcs={Person: str})
        ... def get_person_name(person: Person):
        ...     return person.name

        Alternatively, you can map the type's fully-qualified name
        (e.g. ``"__main__.Person"``) to the hash function instead:

        >>> import streamlit as st
        >>> from pydantic import BaseModel
        >>>
        >>> class Person(BaseModel):
        ...     name: str
        >>>
        >>> @st.cache_resource(hash_funcs={"__main__.Person": str})
        ... def get_person_name(person: Person):
        ...     return person.name
        """
        self._maybe_show_deprecation_warning()
        if func is None:
            return lambda f: make_cached_func_wrapper(CachedResourceFuncInfo(func=f, show_spinner=show_spinner, max_entries=max_entries, ttl=ttl, validate=validate, allow_widgets=experimental_allow_widgets, hash_funcs=hash_funcs))
        return make_cached_func_wrapper(CachedResourceFuncInfo(func=cast(types.FunctionType, func), show_spinner=show_spinner, max_entries=max_entries, ttl=ttl, validate=validate, allow_widgets=experimental_allow_widgets, hash_funcs=hash_funcs))

    @gather_metrics('clear_resource_caches')
    def clear(self) -> None:
        """Clear all cache_resource caches."""
        self._maybe_show_deprecation_warning()
        _resource_caches.clear_all()

    def _maybe_show_deprecation_warning(self):
        """If the API is being accessed with the deprecated `st.experimental_singleton` name,
        show a deprecation warning.
        """
        if self._deprecation_warning is not None:
            show_deprecation_warning(self._deprecation_warning)