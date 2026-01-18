from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import copy
import datetime
import io
import os.path
import shutil
import tarfile
import uuid
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudbuild import snapshot
from googlecloudsdk.api_lib.clouddeploy import client_util
from googlecloudsdk.api_lib.clouddeploy import delivery_pipeline
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.code.cloud import cloudrun
from googlecloudsdk.command_lib.deploy import deploy_util
from googlecloudsdk.command_lib.deploy import exceptions
from googlecloudsdk.command_lib.deploy import rollout_util
from googlecloudsdk.command_lib.deploy import staging_bucket_util
from googlecloudsdk.command_lib.deploy import target_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.resource import yaml_printer
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def ListCurrentDeployedTargets(release_ref, targets):
    """Lists the targets where the given release is the latest.

  Args:
    release_ref: protorpc.messages.Message, protorpc.messages.Message, release
      reference.
    targets: protorpc.messages.Message, protorpc.messages.Message, list of
      target objects.

  Returns:
    A list of target references where this release is deployed.
  """
    matching_targets = []
    release_name = release_ref.RelativeName()
    pipeline_ref = release_ref.Parent()
    for obj in targets:
        target_name = obj.name
        target_ref = target_util.TargetReferenceFromName(target_name)
        rollout_obj = target_util.GetCurrentRollout(target_ref, pipeline_ref)
        if rollout_obj is None:
            continue
        rollout_ref = rollout_util.RolloutReferenceFromName(rollout_obj.name)
        deployed_release_name = rollout_ref.Parent().RelativeName()
        if release_name == deployed_release_name:
            matching_targets.append(target_ref)
    return matching_targets