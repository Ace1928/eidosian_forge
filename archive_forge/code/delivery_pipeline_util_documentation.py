from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import uuid
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.clouddeploy import client_util
from googlecloudsdk.api_lib.clouddeploy import delivery_pipeline
from googlecloudsdk.command_lib.deploy import exceptions as cd_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
Creates a rollback rollout for the target based on the given inputs.

  Args:
    pipeline_rel_name: delivery_pipeline name
    target_id: the target to rollback
    validate_only: whether or not to validate only for the call
    release_id: the release_id to rollback to
    rollout_id: the rollout_id of the new rollout
    rollout_to_rollback: the rollout that is being rolled back by this rollout
    rollout_obj: the rollout resource to pass into rollbackTargetConfig
    starting_phase: starting_phase of the rollout

  Returns:
    RollbackTargetResponse
  