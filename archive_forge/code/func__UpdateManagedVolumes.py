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
def _UpdateManagedVolumes(self, resource, volume_mounts, res_volumes, external_mounts):
    """Update volumes for Cloud Run. Ensure only one secret per directory."""
    new_volumes = {}
    volumes_to_mounts = collections.defaultdict(list)
    for path, vol in volume_mounts.items():
        volumes_to_mounts[vol].append(path)
    for file_path, reachable_secret in self.updates.items():
        mount_point = file_path.rsplit('/', 1)[0]
        if mount_point in new_volumes:
            if new_volumes[mount_point].secretName != reachable_secret.secret_name:
                raise exceptions.ConfigurationError('Cannot update secret at [{}] because a different secret is already mounted in the same directory.'.format(file_path))
            reachable_secret.AppendToSecretVolumeSource(resource, new_volumes[mount_point])
        else:
            new_volumes[mount_point] = reachable_secret.AsSecretVolumeSource(resource)
    for mount_point, volume_source in new_volumes.items():
        if mount_point in volume_mounts:
            volume_name = volume_mounts[mount_point]
            if len(volumes_to_mounts[volume_name]) > 1 or volume_name in external_mounts:
                volumes_to_mounts[volume_name].remove(mount_point)
                new_name = _CopyToNewVolume(resource, volume_name, mount_point, volume_source, res_volumes, volume_mounts)
                volumes_to_mounts[new_name].append(mount_point)
                continue
            else:
                volume = res_volumes[volume_name]
                if volume.secretName != volume_source.secretName:
                    existing_paths = {item.path for item in volume.items}
                    new_paths = {item.path for item in volume_source.items}
                    if not existing_paths.issubset(new_paths):
                        raise exceptions.ConfigurationError('Multiple secrets are specified for directory [{}]. Cloud Run only supports one secret per directory'.format(mount_point))
                else:
                    new_paths = {item.path for item in volume_source.items}
                    for item in volume.items:
                        if item.path not in new_paths:
                            volume_source.items.append(item)
        else:
            volume_name = _UniqueVolumeName(volume_source.secretName, resource.template.volumes)
            try:
                volume_mounts[mount_point] = volume_name
            except KeyError:
                raise exceptions.ConfigurationError('Cannot update mount [{}] because its mounted volume is of a different source type.'.format(mount_point))
        res_volumes[volume_name] = volume_source