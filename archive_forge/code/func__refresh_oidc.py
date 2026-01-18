import atexit
import base64
import copy
import datetime
import json
import logging
import os
import platform
import tempfile
import time
import google.auth
import google.auth.transport.requests
import oauthlib.oauth2
import urllib3
from ruamel import yaml
from requests_oauthlib import OAuth2Session
from six import PY3
from kubernetes.client import ApiClient, Configuration
from kubernetes.config.exec_provider import ExecProvider
from .config_exception import ConfigException
from .dateutil import UTC, format_rfc3339, parse_rfc3339
def _refresh_oidc(self, provider):
    config = Configuration()
    if 'idp-certificate-authority-data' in provider['config']:
        ca_cert = tempfile.NamedTemporaryFile(delete=True)
        if PY3:
            cert = base64.b64decode(provider['config']['idp-certificate-authority-data']).decode('utf-8')
        else:
            cert = base64.b64decode(provider['config']['idp-certificate-authority-data'] + '==')
        with open(ca_cert.name, 'w') as fh:
            fh.write(cert)
        config.ssl_ca_cert = ca_cert.name
    else:
        config.verify_ssl = False
    client = ApiClient(configuration=config)
    response = client.request(method='GET', url='%s/.well-known/openid-configuration' % provider['config']['idp-issuer-url'])
    if response.status != 200:
        return
    response = json.loads(response.data)
    request = OAuth2Session(client_id=provider['config']['client-id'], token=provider['config']['refresh-token'], auto_refresh_kwargs={'client_id': provider['config']['client-id'], 'client_secret': provider['config']['client-secret']}, auto_refresh_url=response['token_endpoint'])
    try:
        refresh = request.refresh_token(token_url=response['token_endpoint'], refresh_token=provider['config']['refresh-token'], auth=(provider['config']['client-id'], provider['config']['client-secret']), verify=config.ssl_ca_cert if config.verify_ssl else None)
    except oauthlib.oauth2.rfc6749.errors.InvalidClientIdError:
        return
    provider['config'].value['id-token'] = refresh['id_token']
    provider['config'].value['refresh-token'] = refresh['refresh_token']