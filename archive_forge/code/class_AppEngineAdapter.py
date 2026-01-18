import requests
import warnings
from requests import adapters
from requests import sessions
from .. import exceptions as exc
from .._compat import gaecontrib
from .._compat import timeout
class AppEngineAdapter(AppEngineMROHack, adapters.HTTPAdapter):
    """The transport adapter for Requests to use urllib3's GAE support.

    Implements Requests's HTTPAdapter API.

    When deploying to Google's App Engine service, some of Requests'
    functionality is broken. There is underlying support for GAE in urllib3.
    This functionality, however, is opt-in and needs to be enabled explicitly
    for Requests to be able to use it.
    """
    __attrs__ = adapters.HTTPAdapter.__attrs__ + ['_validate_certificate']

    def __init__(self, validate_certificate=True, *args, **kwargs):
        _check_version()
        self._validate_certificate = validate_certificate
        super(AppEngineAdapter, self).__init__(*args, **kwargs)

    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = _AppEnginePoolManager(self._validate_certificate)