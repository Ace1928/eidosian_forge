import datetime
from apitools.gen import message_registry
from apitools.gen import service_registry
from apitools.gen import util
def _ApitoolsVersion():
    """Returns version of the currently installed google-apitools package."""
    try:
        import pkg_resources
    except ImportError:
        return 'X.X.X'
    try:
        return pkg_resources.get_distribution('google-apitools').version
    except pkg_resources.DistributionNotFound:
        return 'X.X.X'