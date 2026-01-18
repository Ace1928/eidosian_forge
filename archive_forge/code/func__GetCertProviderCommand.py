from __future__ import absolute_import
from __future__ import unicode_literals
import json
import os
import bootstrapping
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import config
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import gce as c_gce
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
def _GetCertProviderCommand(context_config):
    """Returns the cert provider command from the context config."""
    if hasattr(context_config, 'cert_provider_command'):
        return context_config.cert_provider_command
    try:
        contents = files.ReadFileContents(context_config.config_path)
        json_out = json.loads(contents)
        if 'cert_provider_command' in json_out:
            return json_out['cert_provider_command']
    except files.Error as e:
        log.debug('context aware settings discovery file %s - %s', context_config.config_path, e)