import requests
import warnings
from requests import adapters
from requests import sessions
from .. import exceptions as exc
from .._compat import gaecontrib
from .._compat import timeout
class AppEngineMROHack(adapters.HTTPAdapter):
    """Resolves infinite recursion when monkeypatching.

    This works by injecting itself as the base class of both the
    :class:`AppEngineAdapter` and Requests' default HTTPAdapter, which needs to
    be done because default HTTPAdapter's MRO is recompiled when we
    monkeypatch, at which point this class becomes HTTPAdapter's base class.
    In addition, we use an instantiation flag to avoid infinite recursion.
    """
    _initialized = False

    def __init__(self, *args, **kwargs):
        if not self._initialized:
            self._initialized = True
            super(AppEngineMROHack, self).__init__(*args, **kwargs)