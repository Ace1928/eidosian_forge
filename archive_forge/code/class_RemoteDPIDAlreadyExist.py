import collections
import logging
import os_ken.exception as os_ken_exc
from os_ken.base import app_manager
from os_ken.controller import event
class RemoteDPIDAlreadyExist(os_ken_exc.OSKenException):
    message = 'port (%(dpid)s, %(port)s) has already remote dpid %(remote_dpid)s'