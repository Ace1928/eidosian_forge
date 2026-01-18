from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import exit_code
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions as core_exceptions
class InvalidTestArgError(TestingError):
    """An invalid/unknown test argument was found in an argument file."""

    def __init__(self, arg_name):
        super(InvalidTestArgError, self).__init__('[{0}] is not a valid argument name for: gcloud test run.'.format(arg_name))