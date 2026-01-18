from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class AutoscalingPolicies(base.Group):
    """Create and manage Dataproc autoscaling policies.

  Create and manage Dataproc autoscaling policies.

  ## EXAMPLES

  To see the list of all autoscaling policies, run:

    $ {command} list

  To view the details of an autoscaling policy, run:

    $ {command} describe my_policy

  To view just the non-output only fields of an autoscaling policy, run:

    $ {command} export my_policy --destination policy-file.yaml

  To create or update an autoscaling policy, run:

    $ {command} import my_policy --source policy-file.yaml

  To delete an autoscaling policy, run:

    $ {command} delete my_policy
  """
    pass