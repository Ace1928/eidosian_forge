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
def _extract_error_json(body, resp):
    """Return error_message from the HTTP response body."""
    try:
        content_type = resp.headers.get('Content-Type', '')
    except AttributeError:
        content_type = ''
    if content_type.startswith('application/json'):
        try:
            body_json = resp.json()
            return _extract_error_json_text(body_json)
        except AttributeError:
            body_json = jsonutils.loads(body)
            return _extract_error_json_text(body_json)
        except ValueError:
            return {}
    else:
        try:
            body_json = jsonutils.loads(body)
            return _extract_error_json_text(body_json)
        except ValueError:
            return {}