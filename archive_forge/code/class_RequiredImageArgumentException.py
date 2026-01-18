from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
from googlecloudsdk.api_lib.util import exceptions as exceptions_util
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.core import exceptions
import six
class RequiredImageArgumentException(c_exceptions.RequiredArgumentException):
    """An exception for missing image flag for containers."""

    def __init__(self, containers):
        super(RequiredImageArgumentException, self).__init__('--image', 'Containers {} require a container image to deploy.'.format(', '.join(containers)))