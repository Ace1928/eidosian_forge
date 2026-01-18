from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.metastore import util as api_util
from googlecloudsdk.calliope import base
Calls the Metastore Federations.Delete method.

  Args:
    relative_resource_name: str, the relative resource name of the Metastore
      federation to delete.
    release_track: base.ReleaseTrack, the release track of command. Will dictate
      which Metastore client library will be used.

  Returns:
    Operation: the operation corresponding to the deletion of the federation.
  