import copy
from http import client as http_client
import io
import logging
import os
import socket
import ssl
from urllib import parse as urlparse
from keystoneauth1 import adapter
from oslo_serialization import jsonutils
from oslo_utils import importutils
from magnumclient import exceptions
def _construct_http_client(*args, **kwargs):
    session = kwargs.pop('session', None)
    auth = kwargs.pop('auth', None)
    if session:
        service_type = kwargs.pop('service_type', 'baremetal')
        interface = kwargs.pop('endpoint_type', None)
        region_name = kwargs.pop('region_name', None)
        return SessionClient(session=session, auth=auth, interface=interface, service_type=service_type, region_name=region_name, service_name=None, user_agent='python-magnumclient')
    else:
        return HTTPClient(*args, **kwargs)