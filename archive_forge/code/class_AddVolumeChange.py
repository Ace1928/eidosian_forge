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
class AddVolumeChange(TemplateConfigChanger):
    """Updates Volumes set on the service or job template.

  Attributes:
    new_volumes: The volumes to add.
    release_track: The resource's release track. Used to verify volume types are
      supported in that release track.
  """
    new_volumes: Collection[Mapping[str, str]]
    release_track: base.ReleaseTrack

    def Adjust(self, resource):
        for to_add in self.new_volumes:
            volumes.add_volume(to_add, resource.template.volumes, resource.MessagesModule(), self.release_track)
        return resource