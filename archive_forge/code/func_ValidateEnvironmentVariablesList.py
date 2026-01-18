from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import posixpath
import random
import re
import string
import sys
from googlecloudsdk.api_lib.firebase.test import exceptions as test_exceptions
from googlecloudsdk.api_lib.firebase.test import util as util
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import files
import six
def ValidateEnvironmentVariablesList(args):
    """Validates key-value pairs for 'environment-variables' flag."""
    for key in args.environment_variables or []:
        if not _ENVIRONMENT_VARIABLE_REGEX.match(key):
            raise test_exceptions.InvalidArgException('environment_variables', 'Invalid environment variable [{0}]'.format(key))