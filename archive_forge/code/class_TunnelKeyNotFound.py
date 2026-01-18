import collections
import logging
import os_ken.exception as os_ken_exc
from os_ken.base import app_manager
from os_ken.controller import event
class TunnelKeyNotFound(os_ken_exc.OSKenException):
    message = 'no tunnel key for network %(network_id)s'