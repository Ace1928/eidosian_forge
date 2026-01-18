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
def _CreateGoogleAuthClientConfig(client_id_file=None):
    """Creates a client config from a client id file or gcloud's properties."""
    if client_id_file:
        with files.FileReader(client_id_file) as f:
            return json.load(f)
    return _CreateGoogleAuthClientConfigFromProperties()