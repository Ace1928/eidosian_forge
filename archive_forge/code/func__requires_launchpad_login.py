from urllib.parse import urlsplit
from ... import debug, errors, trace, transport
from ...i18n import gettext
from ...urlutils import InvalidURL, split, join
from .account import get_lp_login
from .uris import DEFAULT_INSTANCE, LAUNCHPAD_DOMAINS, LPNET_SERVICE_ROOT
def _requires_launchpad_login(scheme, netloc, path, query, fragment):
    """Does the URL require a Launchpad login in order to be reached?

    The URL is specified by its parsed components, as returned from
    urlsplit.
    """
    return scheme in ('bzr+ssh', 'sftp', 'git+ssh') and (netloc.endswith('launchpad.net') or netloc.endswith('launchpad.test'))