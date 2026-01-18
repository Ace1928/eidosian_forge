from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
from typing import List, Optional
from googlecloudsdk.command_lib.runapps import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_client
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
def _GetAllTypeMetadata() -> List[TypeMetadata]:
    """Returns metadata for each integration type.

  This loads the metadata from a yaml file at most once and will return the
  same data stored in memory upon future calls.

  Returns:
    array, the type metadata list
  """
    global _TYPE_METADATA
    if _TYPE_METADATA is None:
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'metadata.yaml')
        metadata = yaml.load_path(filename)
        _TYPE_METADATA = [TypeMetadata(**integ) for integ in metadata['integrations']]
    return _TYPE_METADATA