from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.firebase.test import arg_validate
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
import six
def _ValidateArgGroupNames(group_names):
    for group_name in group_names:
        if not _ARG_GROUP_PATTERN.match(group_name):
            raise calliope_exceptions.BadFileException('Invalid argument group name [{0}]. Names may only use a-zA-Z0-9._-'.format(group_name))