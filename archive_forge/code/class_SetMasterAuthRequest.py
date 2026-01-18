from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SetMasterAuthRequest(_messages.Message):
    """SetMasterAuthRequest updates the admin password of a cluster.

  Enums:
    ActionValueValuesEnum: Required. The exact form of action to be taken on
      the master auth.

  Fields:
    action: Required. The exact form of action to be taken on the master auth.
    clusterId: Deprecated. The name of the cluster to upgrade. This field has
      been deprecated and replaced by the name field.
    name: The name (project, location, cluster) of the cluster to set auth.
      Specified in the format `projects/*/locations/*/clusters/*`.
    projectId: Deprecated. The Google Developers Console [project ID or
      project number](https://cloud.google.com/resource-manager/docs/creating-
      managing-projects). This field has been deprecated and replaced by the
      name field.
    update: Required. A description of the update.
    zone: Deprecated. The name of the Google Compute Engine
      [zone](https://cloud.google.com/compute/docs/zones#available) in which
      the cluster resides. This field has been deprecated and replaced by the
      name field.
  """

    class ActionValueValuesEnum(_messages.Enum):
        """Required. The exact form of action to be taken on the master auth.

    Values:
      UNKNOWN: Operation is unknown and will error out.
      SET_PASSWORD: Set the password to a user generated value.
      GENERATE_PASSWORD: Generate a new password and set it to that.
      SET_USERNAME: Set the username. If an empty username is provided, basic
        authentication is disabled for the cluster. If a non-empty username is
        provided, basic authentication is enabled, with either a provided
        password or a generated one.
    """
        UNKNOWN = 0
        SET_PASSWORD = 1
        GENERATE_PASSWORD = 2
        SET_USERNAME = 3
    action = _messages.EnumField('ActionValueValuesEnum', 1)
    clusterId = _messages.StringField(2)
    name = _messages.StringField(3)
    projectId = _messages.StringField(4)
    update = _messages.MessageField('MasterAuth', 5)
    zone = _messages.StringField(6)