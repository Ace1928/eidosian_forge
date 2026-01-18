import requests
import warnings
from requests import adapters
from requests import sessions
from .. import exceptions as exc
from .._compat import gaecontrib
from .._compat import timeout
class InsecureAppEngineAdapter(AppEngineAdapter):
    """An always-insecure GAE adapter for Requests.

    This is a variant of the the transport adapter for Requests to use
    urllib3's GAE support that does not validate certificates. Use with
    caution!

    .. note::
        The ``validate_certificate`` keyword argument will not be honored here
        and is not part of the signature because we always force it to
        ``False``.

    See :class:`AppEngineAdapter` for further details.
    """

    def __init__(self, *args, **kwargs):
        if kwargs.pop('validate_certificate', False):
            warnings.warn('Certificate validation cannot be specified on the InsecureAppEngineAdapter, but was present. This will be ignored and certificate validation will remain off.', exc.IgnoringGAECertificateValidation)
        super(InsecureAppEngineAdapter, self).__init__(*args, validate_certificate=False, **kwargs)