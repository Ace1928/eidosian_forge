import copy
import logging
from urllib import parse
from oslo_serialization import jsonutils
from oslo_utils import importutils
from oslo_utils import strutils
import re
import requests
from manilaclient import exceptions
def _add_log_handlers(self, http_log_debug):
    self._logger = logging.getLogger(__name__)
    if http_log_debug and (not self._logger.handlers):
        ch = logging.StreamHandler()
        ch._name = 'http_client_handler'
        self._logger.setLevel(logging.DEBUG)
        self._logger.addHandler(ch)
        if hasattr(requests, 'logging'):
            rql = requests.logging.getLogger(requests.__name__)
            rql.addHandler(ch)