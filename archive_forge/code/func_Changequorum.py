from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.spanner.v1 import spanner_v1_messages as messages
def Changequorum(self, request, global_params=None):
    """ChangeQuorum is strictly restricted to databases that use dual region instance configurations. Initiates a background operation to change quorum a database from dual-region mode to single-region mode and vice versa. The returned long-running operation will have a name of the format `projects//instances//databases//operations/` and can be used to track execution of the ChangeQuorum. The metadata field type is ChangeQuorumMetadata. Authorization requires `spanner.databases.changequorum` permission on the resource database.

      Args:
        request: (ChangeQuorumRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('Changequorum')
    return self._RunMethod(config, request, global_params=global_params)