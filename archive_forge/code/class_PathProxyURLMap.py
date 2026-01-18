import html
import re
import os
from collections.abc import MutableMapping as DictMixin
from paste import httpexceptions
class PathProxyURLMap(object):
    """
    This is a wrapper for URLMap that catches any strings that
    are passed in as applications; these strings are treated as
    filenames (relative to `base_path`) and are passed to the
    callable `builder`, which will return an application.

    This is intended for cases when configuration files can be
    treated as applications.

    `base_paste_url` is the URL under which all applications added through
    this wrapper must go.  Use ``""`` if you want this to not
    change incoming URLs.
    """

    def __init__(self, map, base_paste_url, base_path, builder):
        self.map = map
        self.base_paste_url = self.map.normalize_url(base_paste_url)
        self.base_path = base_path
        self.builder = builder

    def __setitem__(self, url, app):
        if isinstance(app, str):
            app_fn = os.path.join(self.base_path, app)
            app = self.builder(app_fn)
        url = self.map.normalize_url(url)
        url = (url[0] or self.base_paste_url[0], self.base_paste_url[1] + url[1])
        self.map[url] = app

    def __getattr__(self, attr):
        return getattr(self.map, attr)

    def not_found_application__get(self):
        return self.map.not_found_application

    def not_found_application__set(self, value):
        self.map.not_found_application = value
    not_found_application = property(not_found_application__get, not_found_application__set)