from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import json
import re
from googlecloudsdk.api_lib.container import util as c_util
from googlecloudsdk.command_lib.compute.instance_templates import service_proxy_aux_data
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import kube_util as hub_kube_util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def _GetNestedKeyFromManifest(manifest, *keys):
    """Get the value of a key path from a dict.

  Args:
    manifest: the dict representation of a manifest
    *keys: an ordered list of items in the nested key

  Returns:
    The value of the nested key in the manifest. None, if the nested key does
    not exist.
  """
    for key in keys:
        if not isinstance(manifest, dict):
            return None
        try:
            manifest = manifest[key]
        except KeyError:
            return None
    return manifest