from typing import Any, Dict, Optional
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.compute.base import VolumeSnapshot
class VultrException(Exception):
    """
    Error originating from the Vultr API
    """

    def __init__(self, code, message):
        self.code = code
        self.message = message
        self.args = (code, message)

    def __str__(self):
        return '(%u) %s' % (self.code, self.message)

    def __repr__(self):
        return "VultrException code %u '%s'" % (self.code, self.message)