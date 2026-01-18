import sys
from oslo_serialization import jsonutils
from oslo_utils import reflection
from heatclient._i18n import _
class StackFailure(Exception):
    pass