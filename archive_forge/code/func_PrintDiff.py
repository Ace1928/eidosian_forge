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
def PrintDiff(release_ref, release_obj, target_id=None, prompt=''):
    """Prints differences between current and snapped delivery pipeline and target definitions.

  Args:
    release_ref: protorpc.messages.Message, release resource object.
    release_obj: apitools.base.protorpclite.messages.Message, release message.
    target_id: str, target id, e.g. test/stage/prod.
    prompt: str, prompt text.
  """
    resource_created, resource_changed, resource_not_found = DiffSnappedPipeline(release_ref, release_obj, target_id)
    if resource_created:
        prompt += RESOURCE_CREATED.format('\n'.join(BulletedList(resource_created)))
    if resource_not_found:
        prompt += RESOURCE_NOT_FOUND.format('\n'.join(BulletedList(resource_not_found)))
    if resource_changed:
        prompt += RESOURCE_CHANGED.format('\n'.join(BulletedList(resource_changed)))
    log.status.Print(prompt)