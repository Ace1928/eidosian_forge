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
class ResourceChanges(ContainerConfigChanger):
    """Represents the user intent to update resource limits.

  Attributes:
    memory: Updated memory limit to set in the container. Specified as string
      ending in 'Mi' or 'Gi'. If None the memory limit is not changed.
    cpu: Updated cpu limit to set in the container if not None.
    gpu: Updated gpu limit to set in the container if not None.
  """
    memory: str | None = None
    cpu: str | None = None
    gpu: str | None = None

    def AdjustContainer(self, container, messages_mod):
        """Mutates the given config's resource limits to match what's desired."""
        if self.memory is not None:
            container.resource_limits['memory'] = self.memory
        if self.cpu is not None:
            container.resource_limits['cpu'] = self.cpu
        if self.gpu is not None:
            container.resource_limits['nvidia.com/gpu'] = self.gpu