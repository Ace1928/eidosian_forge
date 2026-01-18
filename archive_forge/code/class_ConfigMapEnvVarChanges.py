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
class ConfigMapEnvVarChanges(TemplateConfigChanger):
    """Represents the user intent to modify environment variable config maps."""

    def __init__(self, updates, removes, clear_others):
        """Initialize a new ConfigMapEnvVarChanges object.

    Args:
      updates: {str, str}, Update env var names and values.
      removes: [str], List of env vars to remove.
      clear_others: bool, If true, clear all non-updated env vars.

    Raises:
      ConfigurationError if a key hasn't been provided for a source.
    """
        super().__init__()
        self._updates = {}
        for name, v in updates.items():
            value = v.split(':', 1)
            if len(value) < 2:
                value.append(self._OmittedSecretKeyDefault(name))
            self._updates[name] = value
        self._removes = removes
        self._clear_others = clear_others

    def _OmittedSecretKeyDefault(self, name):
        if platforms.IsManaged():
            return 'latest'
        raise exceptions.ConfigurationError('Missing required item key for environment variable [{}].'.format(name))

    def Adjust(self, resource):
        """Mutates the given config's env vars to match the desired changes.

    Args:
      resource: k8s_object to adjust

    Returns:
      The adjusted resource

    Raises:
      ConfigurationError if there's an attempt to replace the source of an
        existing environment variable whose source is of a different type
        (e.g. env var's secret source can't be replaced with a config map
        source).
    """
        env_vars = resource.template.env_vars.config_maps
        _PruneMapping(env_vars, self._removes, self._clear_others)
        for name, (source_name, source_key) in self._updates.items():
            try:
                env_vars[name] = self._MakeEnvVarSource(resource.MessagesModule(), source_name, source_key)
            except KeyError:
                raise exceptions.ConfigurationError('Cannot update environment variable [{}] to the given type because it has already been set with a different type.'.format(name))
        return resource

    def _MakeEnvVarSource(self, messages, name, key):
        return messages.EnvVarSource(configMapKeyRef=messages.ConfigMapKeySelector(name=name, key=key))