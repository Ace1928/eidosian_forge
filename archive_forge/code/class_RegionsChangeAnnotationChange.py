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
class RegionsChangeAnnotationChange(NonTemplateConfigChanger):
    """Adds or removes regions annotation on an existing service.

  Attributes:
    existing: the existing Service.
    to_add: A comma-separated list of regions to add to existing.
    to_remove: A comma-separated list of regions to remove from existing.
  """
    to_add: str
    to_remove: str

    def Adjust(self, resource):
        annotation = resource.annotations[k8s_object.MULTI_REGION_REGIONS_ANNOTATION] or None
        existing = set(annotation.split(',') if annotation else [])
        to_add = set(self.to_add.split(',') if self.to_add else [])
        to_remove = set(self.to_remove.split(',') if self.to_remove else [])
        already_added = existing & to_add
        if already_added:
            raise exceptions.ConfigurationError('Multi-region Service already exists in {}'.format(already_added))
        cant_remove = to_remove - (to_remove & existing)
        if cant_remove:
            raise exceptions.ConfigurationError('Multi-region Service not deployed to {}'.format(cant_remove))
        final_list = ','.join((existing | to_add) - to_remove)
        resource.annotations[k8s_object.MULTI_REGION_REGIONS_ANNOTATION] = final_list
        return resource