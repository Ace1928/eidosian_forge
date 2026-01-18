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
class ConfigMapVolumeChanges(TemplateConfigChanger):
    """Represents the user intent to change volumes with config map source types."""

    def __init__(self, updates, removes, clear_others):
        """Initialize a new ConfigMapVolumeChanges object.

    Args:
      updates: {str, [str, str]}, Update mount path and volume fields.
      removes: [str], List of mount paths to remove.
      clear_others: bool, If true, clear all non-updated volumes and mounts of
        the given [volume_type].
    """
        super().__init__()
        self._updates = {}
        for k, v in updates.items():
            update_value = v.split(':', 1)
            if len(update_value) < 2:
                update_value.append(None)
            self._updates[k] = update_value
        self._removes = removes
        self._clear_others = clear_others

    def Adjust(self, resource):
        """Mutates the given config's volumes to match the desired changes.

    Args:
      resource: k8s_object to adjust

    Returns:
      The adjusted resource

    Raises:
      ConfigurationError if there's an attempt to replace the volume a mount
        points to whose existing volume has a source of a different type than
        the new volume (e.g. mount that points to a volume with a secret source
        can't be replaced with a volume that has a config map source).
    """
        volume_mounts = resource.template.container.volume_mounts.config_maps
        res_volumes = resource.template.volumes.config_maps
        _PruneMapping(volume_mounts, self._removes, self._clear_others)
        for path, (source_name, source_key) in self._updates.items():
            volume_name = _UniqueVolumeName(source_name, resource.template.volumes)
            try:
                volume_mounts[path] = volume_name
            except KeyError:
                raise exceptions.ConfigurationError('Cannot update mount [{}] because its mounted volume is of a different source type.'.format(path))
            res_volumes[volume_name] = self._MakeVolumeSource(resource.MessagesModule(), source_name, source_key)
        mounted_volumes = frozenset(itertools.chain.from_iterable((container.volume_mounts.config_maps.values() for container in resource.template.containers.values())))
        _PruneVolumes(mounted_volumes, res_volumes)
        return resource

    def _MakeVolumeSource(self, messages, name, key=None):
        source = messages.ConfigMapVolumeSource(name=name)
        if key is not None:
            source.items.append(messages.KeyToPath(key=key, path=key))
        return source