from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import enum
import itertools
import re
import uuid
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import platforms
def _AssertValidLocalSecretName(self, name):
    if not re.search('^' + self._SECRET_NAME_PARTIAL + '$', name):
        raise exceptions.ConfigurationError('%r is not a valid secret name.' % name)