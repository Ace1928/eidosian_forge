from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import log
from six.moves import filter  # pylint: disable=redefined-builtin
def _IsPublicVersion(operation):
    for o in operation.metadata.additionalProperties:
        if o.key == 'apiVersion':
            return o.value.string_value != 'v1internal1'
    return True