import urllib3
import warnings
from .exceptions import RequestsDependencyWarning
from urllib3.exceptions import DependencyWarning
from .__version__ import __title__, __description__, __url__, __version__
from .__version__ import __build__, __author__, __author_email__, __license__
from .__version__ import __copyright__, __cake__
from . import utils
from . import packages
from .models import Request, Response, PreparedRequest
from .api import request, get, head, post, patch, put, delete, options
from .sessions import session, Session
from .status_codes import codes
from .exceptions import (
import logging
from logging import NullHandler
def _check_cryptography(cryptography_version):
    try:
        cryptography_version = list(map(int, cryptography_version.split('.')))
    except ValueError:
        return
    if cryptography_version < [1, 3, 4]:
        warning = 'Old version of cryptography ({}) may cause slowdown.'.format(cryptography_version)
        warnings.warn(warning, RequestsDependencyWarning)