from __future__ import absolute_import
from __future__ import unicode_literals
import gcloud
import sys
import json
import os
import platform
from googlecloudsdk.core import config
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from six.moves import input
def GetDefaultInstalledComponents():
    """Gets the list of components to install by default.

  Returns:
    list(str), The component ids that should be installed.  It will return []
    if there are no default components, or if there is any error in reading
    the file with the defaults.
  """
    default_components_file = os.path.join(BOOTSTRAPPING_DIR, '.default_components')
    try:
        with open(default_components_file) as f:
            return json.load(f)
    except:
        pass
    return []