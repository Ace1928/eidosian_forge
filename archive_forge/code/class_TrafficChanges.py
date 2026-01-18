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
class TrafficChanges(NonTemplateConfigChanger):
    """Represents the user intent to change a service's traffic assignments.

  Attributes:
    new_percentages: New traffic percentages to set.
    by_tag: Boolean indicating that new traffic percentages are specified by
      tag.
    tags_to_update: Traffic tag values to update.
    tags_to_remove: Traffic tags to remove.
    clear_other_tags: Whether nonupdated tags should be cleared.
  """
    new_percentages: Mapping[str, int]
    by_tag: bool = False
    tags_to_update: Mapping[str, str] = dataclasses.field(default_factory=dict)
    tags_to_remove: Container[str] = dataclasses.field(default_factory=list)
    clear_other_tags: bool = False

    def Adjust(self, resource):
        """Mutates the given service's traffic assignments."""
        if self.tags_to_update or self.tags_to_remove or self.clear_other_tags:
            resource.spec_traffic.UpdateTags(self.tags_to_update, self.tags_to_remove, self.clear_other_tags)
        if self.new_percentages:
            if self.by_tag:
                tag_to_key = resource.spec_traffic.TagToKey()
                percentages = {}
                for tag in self.new_percentages:
                    try:
                        percentages[tag_to_key[tag]] = self.new_percentages[tag]
                    except KeyError:
                        raise exceptions.ConfigurationError('There is no revision tagged with [{}] in the traffic allocation for [{}].'.format(tag, resource.name))
            else:
                percentages = self.new_percentages
            resource.spec_traffic.UpdateTraffic(percentages)
        return resource