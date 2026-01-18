from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class UnsupportedKmsKeyTypeException(exceptions.Error):
    """Indicates that a user is using an unsupported KMS key type."""