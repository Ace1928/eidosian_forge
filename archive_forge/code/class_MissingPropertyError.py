from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class MissingPropertyError(exceptions.Error):
    """Indicates a missing property in an ArgDict flag."""

    def __init__(self, flag_name, property_name):
        message = 'Flag [--{}] is missing the required property [{}]'.format(flag_name, property_name)
        super(MissingPropertyError, self).__init__(message)