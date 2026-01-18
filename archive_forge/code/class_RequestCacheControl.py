from __future__ import annotations
from .mixins import ImmutableDictMixin
from .mixins import UpdateDictMixin
from .. import http
class RequestCacheControl(ImmutableDictMixin, _CacheControl):
    """A cache control for requests.  This is immutable and gives access
    to all the request-relevant cache control headers.

    To get a header of the :class:`RequestCacheControl` object again you can
    convert the object into a string or call the :meth:`to_header` method.  If
    you plan to subclass it and add your own items have a look at the sourcecode
    for that class.

    .. versionchanged:: 2.1.0
        Setting int properties such as ``max_age`` will convert the
        value to an int.

    .. versionadded:: 0.5
       In previous versions a `CacheControl` class existed that was used
       both for request and response.
    """
    max_stale = cache_control_property('max-stale', '*', int)
    min_fresh = cache_control_property('min-fresh', '*', int)
    only_if_cached = cache_control_property('only-if-cached', None, bool)