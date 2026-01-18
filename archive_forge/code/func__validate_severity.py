import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
def _validate_severity(severity):
    if severity.upper() not in severity_types:
        errmsg = 'Invalid severity, not one of [' + ', '.join(severity_types) + ']'
        print(errmsg)
        return False
    return True