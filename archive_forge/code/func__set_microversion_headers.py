import datetime
import functools
import hashlib
import json
import logging
import os
import platform
import socket
import sys
import time
import urllib
import uuid
import requests
import keystoneauth1
from keystoneauth1 import _utils as utils
from keystoneauth1 import discover
from keystoneauth1 import exceptions
@staticmethod
def _set_microversion_headers(headers, microversion, service_type, endpoint_filter):
    microversion = discover.normalize_version_number(microversion)
    if microversion[0] != discover.LATEST and discover.LATEST in microversion[1:]:
        raise TypeError("Specifying a '{major}.latest' microversion is not allowed.")
    microversion = discover.version_to_string(microversion)
    if not service_type:
        if endpoint_filter and 'service_type' in endpoint_filter:
            service_type = endpoint_filter['service_type']
        else:
            raise TypeError('microversion {microversion} was requested but no service_type information is available. Either provide a service_type in endpoint_filter or pass microversion_service_type as an argument.'.format(microversion=microversion))
    if service_type.startswith('volume') or service_type == 'block-storage':
        service_type = 'volume'
    elif service_type.startswith('share'):
        service_type = 'shared-file-system'
    headers.setdefault('OpenStack-API-Version', '{service_type} {microversion}'.format(service_type=service_type, microversion=microversion))
    header_names = _mv_legacy_headers_for_service(service_type)
    for h in header_names:
        headers.setdefault(h, microversion)