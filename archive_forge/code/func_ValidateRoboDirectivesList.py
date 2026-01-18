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
def ValidateRoboDirectivesList(args):
    """Validates key-value pairs for 'robo_directives' flag."""
    resource_names = set()
    duplicates = set()
    for key, value in six.iteritems(args.robo_directives or {}):
        action_type, resource_name = util.ParseRoboDirectiveKey(key)
        if action_type in ['click', 'ignore'] and value:
            raise test_exceptions.InvalidArgException('robo_directives', 'Input value not allowed for click or ignore actions: [{0}={1}]'.format(key, value))
        if not resource_name:
            raise test_exceptions.InvalidArgException('robo_directives', 'Missing resource_name for key [{0}].'.format(key))
        if resource_name in resource_names:
            duplicates.add(resource_name)
        else:
            resource_names.add(resource_name)
    if duplicates:
        raise test_exceptions.InvalidArgException('robo_directives', 'Duplicate resource names are not allowed: [{0}].'.format(', '.join(duplicates)))