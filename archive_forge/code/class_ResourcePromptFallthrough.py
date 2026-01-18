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
class ResourcePromptFallthrough(PromptFallthrough):
    """Fall through to reading the resource name from an interactive prompt."""

    def __init__(self, resource_type_lower):
        super(ResourcePromptFallthrough, self).__init__('specify the {} name from an interactive prompt'.format(resource_type_lower))
        self.resource_type_lower = resource_type_lower

    def _Prompt(self, parsed_args):
        message = self.resource_type_lower.capitalize() + ' name'
        default_name = self._DefaultNameFromArgs(parsed_args)
        return console_io.PromptWithDefault(message=message, default=default_name)

    def _DefaultNameFromArgs(self, parsed_args):
        if getattr(parsed_args, 'image', None):
            return _GenerateServiceName(parsed_args.image)
        elif getattr(parsed_args, 'source', None):
            return _GenerateServiceNameFromLocalPath(parsed_args.source)
        return ''