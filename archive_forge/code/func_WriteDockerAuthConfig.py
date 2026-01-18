from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
import os
import subprocess
import sys
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.credentials import store
from googlecloudsdk.core.docker import client_lib
from googlecloudsdk.core.docker import constants
from googlecloudsdk.core.util import files
import six
def WriteDockerAuthConfig(structure):
    """Write out a complete set of Docker authorization entries.

  This is public only to facilitate testing.

  Args:
    structure: The dict of authorization mappings to write to the
               Docker configuration file.
  """
    path, is_new_format = client_lib.GetDockerConfigPath()
    contents = client_lib.ReadConfigurationFile(path)
    if is_new_format:
        full_cfg = contents
        full_cfg['auths'] = structure
        file_contents = json.dumps(full_cfg, indent=2)
    else:
        file_contents = json.dumps(structure, indent=2)
    files.WriteFileAtomically(path, file_contents)