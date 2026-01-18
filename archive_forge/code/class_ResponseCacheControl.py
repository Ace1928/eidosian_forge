from __future__ import annotations
from .mixins import ImmutableDictMixin
from .mixins import UpdateDictMixin
from .. import http
class ResponseCacheControl(_CacheControl):
    """A cache control for responses.  Unlike :class:`RequestCacheControl`
    this is mutable and gives access to response-relevant cache control
    headers.

    To get a header of the :class:`ResponseCacheControl` object again you can
    convert the object into a string or call the :meth:`to_header` method.  If
    you plan to subclass it and add your own items have a look at the sourcecode
    for that class.

    .. versionchanged:: 2.1.1
        ``s_maxage`` converts the value to an int.

    .. versionchanged:: 2.1.0
        Setting int properties such as ``max_age`` will convert the
        value to an int.

    .. versionadded:: 0.5
       In previous versions a `CacheControl` class existed that was used
       both for request and response.
    """
    public = cache_control_property('public', None, bool)
    private = cache_control_property('private', '*', None)
    must_revalidate = cache_control_property('must-revalidate', None, bool)
    proxy_revalidate = cache_control_property('proxy-revalidate', None, bool)
    s_maxage = cache_control_property('s-maxage', None, int)
    immutable = cache_control_property('immutable', None, bool)