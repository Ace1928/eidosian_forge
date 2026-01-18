from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.command_lib.emulators import datastore_util
from googlecloudsdk.command_lib.emulators import pubsub_util
from googlecloudsdk.core.util import files
import six
def WriteJsonToFile(self, output_file):
    """Writes configuration to file.

    The format will be
    {"localEmulators": {emulator1: port1, emulator2: port2},
     "proxyPort": port,
     "shouldProxyToGcp": bool}

    Args:
      output_file: str, file to write to
    """
    data = {'localEmulators': self._local_emulators, 'proxyPort': self._proxy_port, 'shouldProxyToGcp': self._should_proxy_to_gcp}
    files.WriteFileContents(output_file, json.dumps(data, indent=2))