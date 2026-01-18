from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import copy
import os
import re
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.calliope.concepts import util as concepts_util
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
class DefaultFallthrough(deps.Fallthrough):
    """Use the namespace "default".

  For Knative only.

  For Cloud Run, raises an ArgumentError if project not set.
  """

    def __init__(self):
        super(DefaultFallthrough, self).__init__(function=None, hint='For Cloud Run on Kubernetes Engine, defaults to "default". Otherwise, defaults to project ID.')

    def _Call(self, parsed_args):
        if platforms.GetPlatform() == platforms.PLATFORM_GKE or platforms.GetPlatform() == platforms.PLATFORM_KUBERNETES:
            return 'default'
        elif not (getattr(parsed_args, 'project', None) or properties.VALUES.core.project.Get()):
            raise exceptions.ArgumentError('The [project] resource is not properly specified. Please specify the argument [--project] on the command line or set the property [core/project].')
        return None