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
class LabelChanges(ConfigChanger):
    """Represents the user intent to modify metadata labels.

  Attributes:
    diff: Label diff to apply.
    copy_to_revision: A boolean indicating that label changes should be copied
      to the resource's template.
  """
    _LABELS_NOT_ALLOWED_IN_REVISION: ClassVar[frozenset[str]] = frozenset([service.ENDPOINT_VISIBILITY])
    diff: labels_util.Diff
    copy_to_revision: bool = True

    @property
    def adjusts_template(self):
        return self.copy_to_revision

    def Adjust(self, resource):
        update_result = self.diff.Apply(k8s_object.Meta(resource.MessagesModule()).LabelsValue, resource.metadata.labels)
        maybe_new_labels = update_result.GetOrNone()
        if maybe_new_labels:
            resource.metadata.labels = maybe_new_labels
            template = resource.execution_template if hasattr(resource, 'execution_template') else resource.template
            if self.copy_to_revision and hasattr(template, 'labels'):
                nonce = template.labels.get(revision.NONCE_LABEL)
                template.metadata.labels = copy.deepcopy(maybe_new_labels)
                for label_to_remove in self._LABELS_NOT_ALLOWED_IN_REVISION:
                    if label_to_remove in template.labels:
                        del template.labels[label_to_remove]
                if nonce:
                    template.labels[revision.NONCE_LABEL] = nonce
        return resource