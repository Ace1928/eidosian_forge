import abc
from oslo_serialization import jsonutils
import six
def _isset(self, attr):
    """Check to see if attribute is defined."""
    try:
        if isinstance(getattr(self, attr), ValidatorDescriptor):
            return False
        return True
    except AttributeError:
        return False