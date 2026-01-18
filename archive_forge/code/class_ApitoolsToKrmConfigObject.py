from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
import os
from apitools.base.protorpclite import messages
from googlecloudsdk.command_lib.anthos.common import file_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.util import files
import six
class ApitoolsToKrmConfigObject(file_parsers.YamlConfigObject):
    """Abstraction for an Apitools to KRM Mapping file object."""

    def __init__(self, content):
        if not isinstance(content, dict):
            raise file_parsers.YamlConfigObjectError('Invalid ApitoolsToKrmFieldDescriptor content')
        self._apitools_request = list(content.keys())[0]
        self._content = content[self._apitools_request]

    @property
    def apitools_request(self):
        return self._apitools_request

    def __str__(self):
        return '{}:\n{}'.format(self.apitools_request, super(ApitoolsToKrmConfigObject, self).__str__())