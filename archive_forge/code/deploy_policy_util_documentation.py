from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.clouddeploy import deploy_policy
from googlecloudsdk.core import resources
Creates deploy policy canonical resource names from ids.

  Args:
    pipeline_ref: pipeline resource reference.
    deploy_policy_ids: list of deploy policy ids (e.g. ['deploy-policy-1',
      'deploy-policy-2'])

  Returns:
    A list of deploy policy canonical resource names.
  