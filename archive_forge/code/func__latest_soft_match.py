import copy
import re
import urllib
import os_service_types
from keystoneauth1 import _utils as utils
from keystoneauth1 import exceptions
def _latest_soft_match(required, candidate):
    if not required:
        return False
    if LATEST not in required:
        return False
    if all((part == LATEST for part in required)):
        return True
    if required[0] == candidate[0] and required[1] == LATEST:
        return True
    return False