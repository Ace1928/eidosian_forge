from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import argparse
import collections
from collections.abc import Collection, Container, Iterable, Mapping, MutableMapping
import copy
import dataclasses
import itertools
import json
import types
from typing import Any, ClassVar
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import job
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.api_lib.run import revision
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import name_generator
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.command_lib.run import secrets_mapping
from googlecloudsdk.command_lib.run import volumes
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import repeated
import six
@dataclasses.dataclass(frozen=True)
class ReplaceJobChange(NonTemplateConfigChanger):
    """Represents the user intent to replace the job.

  Attributes:
    new_job: New job that will replace the existing job.
  """
    new_job: job.Job

    def Adjust(self, resource):
        """Returns a replacement for resource.

    The returned job is the job provided to the constructor. If
    resource.metadata.resourceVersion is not empty, has metadata.resourceVersion
    of returned job set to this value.

    Args:
      resource: job.Job, The job to adjust.
    """
        if resource.metadata.resourceVersion:
            self.new_job.metadata.resourceVersion = resource.metadata.resourceVersion
        return self.new_job