from googlecloudsdk.command_lib.concepts import concept_managers
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import io
import re
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.command_lib.concepts import base
from googlecloudsdk.command_lib.concepts import dependency_managers
from googlecloudsdk.command_lib.concepts import exceptions
from googlecloudsdk.command_lib.concepts import names
from googlecloudsdk.core.util import scaled_integer
from googlecloudsdk.core.util import semver
from googlecloudsdk.core.util import times
import six
def _ConvertEndpoint(self, endpoint, kind):
    """Declaration time endpoint conversion check."""
    message = None
    try:
        endpoint.value = self._convert_endpoint(endpoint.string)
        return
    except exceptions.ParseError as e:
        message = six.text_type(e).split('. ', 1)[1].rstrip('.')
    except (AttributeError, ValueError) as e:
        message = _SubException(e)
    raise exceptions.ConstraintError(self.GetPresentationName(), kind, endpoint.string, message + '.')