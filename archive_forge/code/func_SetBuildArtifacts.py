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
def SetBuildArtifacts(images, messages, release_config):
    """Set build_artifacts field of the release message.

  Args:
    images: dict[str,dict], docker image name and tag dictionary.
    messages: Module containing the Cloud Deploy messages.
    release_config: apitools.base.protorpclite.messages.Message, Cloud Deploy
      release message.

  Returns:
    Cloud Deploy release message.
  """
    if not images:
        return release_config
    build_artifacts = []
    for key, value in sorted(six.iteritems(images)):
        build_artifacts.append(messages.BuildArtifact(image=key, tag=value))
    release_config.buildArtifacts = build_artifacts
    return release_config