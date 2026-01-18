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
class IngressContainerBaseImagesAnnotationChange(BaseImagesAnnotationChange):
    """Represents the user intent to update the 'base-images' template annotation.

  The value of the annotation is a string representation of a json map of
  container_name -> base_image_url. E.g.: '{"mycontainer":"my_base_image_url"}'.

  This class changes the base image annotation for the default container, which
  is either the container in a service with one container or the one with a port
  set in a service with multiple containers.

  Attributes:
    base_image: url of the base image for the default container or None
  """
    base_image: str | None = None

    def Adjust(self, resource: revision.Revision):
        """Updates the revision to use automatic base image updates."""
        if self.base_image:
            self.updates[resource.template.container.name or ''] = self.base_image
        else:
            self.deletes.append(resource.template.container.name or '')
        return super().Adjust(resource)