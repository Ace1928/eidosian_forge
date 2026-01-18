from typing import cast
from urllib.parse import quote as urlquote, unquote as urlunquote, urlunsplit
from hyperlink import URL as _URL
def _reconstitute(self):
    """
        Reconstitute this L{URLPath} from all its given attributes.
        """
    urltext = urlquote(urlunsplit((self._scheme, self._netloc, self._path, self._query, self._fragment)), safe=_allascii)
    self._url = _URL.fromText(urltext.encode('ascii').decode('ascii'))