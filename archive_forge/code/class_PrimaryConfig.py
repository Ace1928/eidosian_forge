from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrimaryConfig(_messages.Message):
    """Configuration for the primary cluster. It has the list of clusters that
  are replicating from this cluster. This should be set if and only if the
  cluster is of type PRIMARY.

  Fields:
    secondaryClusterNames: Output only. Names of the clusters that are
      replicating from this cluster.
  """
    secondaryClusterNames = _messages.StringField(1, repeated=True)