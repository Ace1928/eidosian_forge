from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import os
import re
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.updater import schemas
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import requests
import six
@staticmethod
def FromInstallState(install_state):
    """Loads a snapshot from the local installation state.

    This creates a snapshot that may not have actually existed at any point in
    time.  It does, however, exactly reflect the current state of your local
    SDK.

    Args:
      install_state: install_state.InstallState, The InstallState object to load
        from.

    Returns:
      A ComponentSnapshot object.
    """
    installed = install_state.InstalledComponents()
    components = [manifest.ComponentDefinition() for manifest in installed.values()]
    sdk_definition = schemas.SDKDefinition(revision=-1, schema_version=None, release_notes_url=None, version=None, gcloud_rel_path=None, post_processing_command=None, components=components, notifications={})
    return ComponentSnapshot(sdk_definition)