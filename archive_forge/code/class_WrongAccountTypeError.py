from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class WrongAccountTypeError(exceptions.Error):
    """Raised when audiences are specified but account type is not service account."""