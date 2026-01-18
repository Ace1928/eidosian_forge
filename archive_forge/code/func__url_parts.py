import copy
import json
import urllib.parse
import requests
@property
def _url_parts(self):
    if self._url_parts_ is None:
        url = self._request.url
        if not self._case_sensitive:
            url = url.lower()
        self._url_parts_ = urllib.parse.urlparse(url)
    return self._url_parts_