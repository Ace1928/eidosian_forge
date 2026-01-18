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
class ReplaceServiceChange(NonTemplateConfigChanger):
    """Represents the user intent to replace the service.

  Attributes:
    new_service: New service that will replace the existing service.
  """
    new_service: service.Service

    def Adjust(self, resource):
        """Returns a replacement for resource.

    The returned service is the service provided to the constructor. If
    resource.metadata.resourceVersion is not empty, has metadata.resourceVersion
    of returned service set to this value.

    Args:
      resource: service.Service, The service to adjust.
    """
        if resource.metadata.resourceVersion:
            self.new_service.metadata.resourceVersion = resource.metadata.resourceVersion
            for k, v in resource.annotations.items():
                if k.startswith(k8s_object.SERVING_GROUP):
                    self.new_service.annotations[k] = v
        return self.new_service