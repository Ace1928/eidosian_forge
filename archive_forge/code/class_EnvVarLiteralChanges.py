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
class EnvVarLiteralChanges(ContainerConfigChanger):
    """Represents the user intent to modify environment variables string literals.

  Attributes:
    updates: Updated env var names and values to set.
    removes: Env vars to remove.
    clear_others: If true clear all non-updated env vars.
  """
    updates: Mapping[str, str] = dataclasses.field(default_factory=dict)
    removes: Collection[str] = dataclasses.field(default_factory=list)
    clear_others: bool = False

    def AdjustContainer(self, container, messages_mod):
        """Mutates the given config's env vars to match the desired changes.

    Args:
      container: container to adjust
      messages_mod: messages module

    Returns:
      The adjusted container

    Raises:
      ConfigurationError if there's an attempt to replace the source of an
        existing environment variable whose source is of a different type
        (e.g. env var's secret source can't be replaced with a config map
        source).
    """
        _PruneMapping(container.env_vars.literals, self.removes, self.clear_others)
        try:
            container.env_vars.literals.update(self.updates)
        except KeyError as e:
            raise exceptions.ConfigurationError('Cannot update environment variable [{}] to string literal because it has already been set with a different type.'.format(e.args[0]))