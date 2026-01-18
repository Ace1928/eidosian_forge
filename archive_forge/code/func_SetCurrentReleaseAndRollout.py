from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.command_lib.deploy import delivery_pipeline_util
from googlecloudsdk.command_lib.deploy import rollout_util
from googlecloudsdk.command_lib.deploy import target_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def SetCurrentReleaseAndRollout(current_rollout, output):
    """Set current release and the last deployment section in the output.

  Args:
    current_rollout: protorpc.messages.Message, rollout object.
    output: dictionary object

  Returns:
    The modified output object with the rollout's parent release, the name of
    the rollout, and the time it was deployed.

  """
    if current_rollout:
        current_rollout_ref = resources.REGISTRY.Parse(current_rollout.name, collection='clouddeploy.projects.locations.deliveryPipelines.releases.rollouts')
        output['Latest release'] = current_rollout_ref.Parent().RelativeName()
        output['Latest rollout'] = current_rollout_ref.RelativeName()
        output['Deployed'] = current_rollout.deployEndTime
    return output