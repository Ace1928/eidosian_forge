from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.filestore import filestore_client
from googlecloudsdk.command_lib.filestore import update_util
from googlecloudsdk.command_lib.filestore import util
from googlecloudsdk.core import properties
def UpdateLabels(ref, args, req):
    """Update snapshot labels."""
    del ref
    new_labels = update_util.GetUpdatedLabels(args, req, update_util.snapshot_feature_name)
    if new_labels:
        req.snapshot.labels = new_labels
    return req