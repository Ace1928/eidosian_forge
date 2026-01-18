import socket
import re
import logging
import warnings
from requests.exceptions import RequestException, SSLError
import http.client as http_client
from urllib.parse import quote, unquote
from urllib.parse import urljoin, urlparse, urlunparse
from time import sleep, time
from swiftclient import version as swiftclient_version
from swiftclient.exceptions import ClientException
from swiftclient.requests_compat import SwiftClientRequestsSession
from swiftclient.utils import (
def _add_response_dict(self, target_dict, kwargs):
    if target_dict is not None and 'response_dict' in kwargs:
        response_dict = kwargs['response_dict']
        if 'response_dicts' in target_dict:
            target_dict['response_dicts'].append(response_dict)
        else:
            target_dict['response_dicts'] = [response_dict]
        target_dict.update(response_dict)