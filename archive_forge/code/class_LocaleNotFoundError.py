from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import exit_code
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions as core_exceptions
class LocaleNotFoundError(TestingError):
    """Failed to find a locale in the test environment catalog."""

    def __init__(self, locale):
        super(LocaleNotFoundError, self).__init__("'{l}' is not a valid locale".format(l=locale))