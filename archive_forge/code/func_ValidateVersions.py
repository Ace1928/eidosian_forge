from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import random
import re
import string
import sys
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.managed_instance_groups import auto_healing_utils
from googlecloudsdk.command_lib.compute.managed_instance_groups import update_instances_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
from six.moves import range  # pylint: disable=redefined-builtin
def ValidateVersions(igm_info, new_versions, resources, force=False):
    """Validates whether versions provided by user are consistent.

  Args:
    igm_info: instance group manager resource.
    new_versions: list of new versions.
    force: if true, we allow any combination of instance templates, as long as
      they are different. If false, only the following transitions are allowed:
      X -> Y, X -> (X, Y), (X, Y) -> X, (X, Y) -> Y, (X, Y) -> (X, Y)

  Raises:
     InvalidArgumentError: if provided arguments are not complete or invalid.
  """
    if len(new_versions) == 2 and new_versions[0].instanceTemplate == new_versions[1].instanceTemplate:
        raise InvalidArgumentError('Provided instance templates must be different.')
    if force:
        return
    if igm_info.versions:
        igm_templates = [resources.ParseURL(version.instanceTemplate).RelativeName() for version in igm_info.versions]
    elif igm_info.instanceTemplate:
        igm_templates = [resources.ParseURL(igm_info.instanceTemplate).RelativeName()]
    else:
        raise InvalidArgumentError('Either versions or instance template must be specified for managed instance group.')
    new_templates = [resources.ParseURL(version.instanceTemplate).RelativeName() for version in new_versions]
    version_count = len(_GetInstanceTemplatesSet(igm_templates, new_templates))
    if version_count > 2:
        raise InvalidArgumentError('Update inconsistent with current state. The only allowed transitions between versions are: X -> Y, X -> (X, Y), (X, Y) -> X, (X, Y) -> Y, (X, Y) -> (X, Y). Please check versions templates or use --force.')