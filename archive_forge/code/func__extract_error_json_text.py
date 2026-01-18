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
def _extract_error_json_text(body_json):
    error_json = {}
    if 'error_message' in body_json:
        raw_msg = body_json['error_message']
        error_json = jsonutils.loads(raw_msg)
    elif 'error' in body_json:
        error_body = body_json['error']
        error_json = {'faultstring': error_body['title'], 'debuginfo': error_body['message']}
    else:
        error_body = body_json['errors'][0]
        error_json = {'faultstring': error_body['title']}
        if 'detail' in error_body:
            error_json['debuginfo'] = error_body['detail']
        elif 'description' in error_body:
            error_json['debuginfo'] = error_body['description']
    return error_json