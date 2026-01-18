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
def _mv_legacy_headers_for_service(mv_service_type):
    """Workaround for services that predate standardization.

    TODO(sdague): eventually convert this to using os-service-types
    and put the logic there. However, right now this is so little
    logic, inlining it for release is a better call.

    """
    headers = []
    if mv_service_type == 'compute':
        headers.append('X-OpenStack-Nova-API-Version')
    elif mv_service_type == 'baremetal':
        headers.append('X-OpenStack-Ironic-API-Version')
    elif mv_service_type in ['sharev2', 'shared-file-system']:
        headers.append('X-OpenStack-Manila-API-Version')
    return headers