from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import json
import textwrap
from googlecloudsdk.command_lib.util import check_browser
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
def _CreateGoogleAuthClientConfigFromProperties():
    """Creates a client config from gcloud's properties."""
    auth_uri = properties.VALUES.auth.auth_host.Get(required=True)
    token_uri = GetTokenUri()
    client_id = properties.VALUES.auth.client_id.Get(required=True)
    client_secret = properties.VALUES.auth.client_secret.Get(required=True)
    return {'installed': {'client_id': client_id, 'client_secret': client_secret, 'auth_uri': auth_uri, 'token_uri': token_uri}}