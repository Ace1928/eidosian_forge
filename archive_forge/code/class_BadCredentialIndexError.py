from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudiot import devices
from googlecloudsdk.api_lib.cloudiot import registries
from googlecloudsdk.command_lib.iot import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import http_encoding
from googlecloudsdk.core.util import times
import six
class BadCredentialIndexError(exceptions.Error):
    """Indicates that a user supplied a bad index for resource's credentials."""

    def __init__(self, name, credentials, index, resource='device'):
        super(BadCredentialIndexError, self).__init__('Invalid credential index [{index}]; {resource} [{name}] has {num_credentials} credentials. (Indexes are zero-based.))'.format(index=index, name=name, num_credentials=len(credentials), resource=resource))