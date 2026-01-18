from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import exit_code
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions as core_exceptions
class InvalidDimensionNameError(TestingError):
    """An invalid test matrix dimension name was encountered."""

    def __init__(self, dim_name):
        super(InvalidDimensionNameError, self).__init__("'{d}' is not a valid dimension name. Must be one of: ['model', 'version', 'locale', 'orientation']".format(d=dim_name))