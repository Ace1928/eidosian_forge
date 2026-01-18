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
class RemoveVolumeMountChange(ContainerConfigChanger):
    """Removes Volume Mounts from the container.

  Attributes:
    removed_mounts: Volume mounts to remove from the adjusted container.
  """
    removed_mounts: Collection[str] = dataclasses.field(default_factory=list)
    clear_mounts: bool = False

    def AdjustContainer(self, container, messages_mod):
        if self.clear_mounts:
            keys = list(container.volume_mounts)
            for mount in keys:
                del container.volume_mounts[mount]
        else:
            for to_remove in self.removed_mounts:
                if to_remove in container.volume_mounts:
                    del container.volume_mounts[to_remove]
        return container