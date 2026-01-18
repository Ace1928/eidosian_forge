import copy
from oslo_serialization import jsonutils
from urllib import parse
from saharaclient._i18n import _
class NotUpdated(object):
    """A sentinel class to signal that parameter should not be updated."""

    def __repr__(self):
        return 'NotUpdated'