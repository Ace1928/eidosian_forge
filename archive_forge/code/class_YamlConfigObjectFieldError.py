from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import copy
import io
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
class YamlConfigObjectFieldError(YamlConfigObjectError):
    """Raised when an invalid field is used on  a YamlConfigObject."""

    def __init__(self, field, object_type, custom_message=None):
        error_msg = 'Invalid field [{}] for YamlConfigObject type [{}]'.format(field, object_type)
        if custom_message:
            error_msg = '{}: {}'.format(error_msg, custom_message)
        super(YamlConfigObjectFieldError, self).__init__(error_msg)