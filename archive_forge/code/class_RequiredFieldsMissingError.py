from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class RequiredFieldsMissingError(exceptions.Error):
    """Error for when calling a method when a required field is unspecified."""