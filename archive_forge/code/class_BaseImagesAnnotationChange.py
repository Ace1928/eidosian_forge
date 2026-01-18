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
class BaseImagesAnnotationChange(TemplateConfigChanger):
    """Represents the user intent to update the 'base-images' template annotation.

  The value of the annotation is a string representation of a json map of
  container_name -> base_image_url. E.g.: '{"mycontainer":"my_base_image_url"}'.

  Attributes:
    updates: {container:url} map of values that need to be added/updated
    deletes: List of containers whose base image url needs to be deleted.
  """
    updates: dict[str, str] = dataclasses.field(default_factory=dict)
    deletes: list[str] = dataclasses.field(default_factory=list)

    def _mergeBaseImageUrls(self, resource: revision.Revision, existing_base_image_urls: dict[str, str], updates: dict[str, str], deletes: list[str]):
        if deletes:
            for container in deletes:
                if container in existing_base_image_urls:
                    del existing_base_image_urls[container]
        if updates:
            for container, url in updates.items():
                existing_base_image_urls[container] = url
        return self._constructBaseImageUrls(resource, existing_base_image_urls)

    def _constructBaseImageUrls(self, resource: revision.Revision, urls: dict[str, str]):
        containers = frozenset([x or '' for x in resource.template.containers.keys()])
        base_images_str = ', '.join((f'"{key}":"{value}"' for key, value in urls.items() if key in containers))
        return '{' + base_images_str + '}' if base_images_str else ''

    def Adjust(self, resource: revision.Revision):
        """Updates the revision to use automatic base image updates."""
        annotations = resource.template.annotations
        existing_value = annotations.get(revision.BASE_IMAGES_ANNOTATION, '')
        if existing_value:
            existing_base_image_urls = json.loads(existing_value)
            new_value = self._mergeBaseImageUrls(resource, existing_base_image_urls, self.updates, self.deletes)
        else:
            new_value = self._constructBaseImageUrls(resource, self.updates)
        if new_value:
            resource.template.annotations[revision.BASE_IMAGES_ANNOTATION] = new_value
            resource.template.spec.runtimeClassName = revision.BASE_IMAGE_UPDATE_RUNTIME_CLASS_NAME
        elif revision.BASE_IMAGES_ANNOTATION in annotations:
            del resource.template.annotations[revision.BASE_IMAGES_ANNOTATION]
            if resource.template.spec.runtimeClassName == revision.BASE_IMAGE_UPDATE_RUNTIME_CLASS_NAME:
                resource.template.spec.runtimeClassName = ''
        return resource