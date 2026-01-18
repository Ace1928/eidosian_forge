import copy
import logging
from urllib import parse
from oslo_serialization import jsonutils
from oslo_utils import importutils
from oslo_utils import strutils
import re
import requests
from manilaclient import exceptions
def _set_request_options(self, insecure, cacert, timeout=None, cert=None):
    options = {'verify': True}
    if insecure:
        options['verify'] = False
    elif cacert:
        options['verify'] = cacert
    if cert:
        options['cert'] = cert
    if timeout:
        options['timeout'] = timeout
    return options