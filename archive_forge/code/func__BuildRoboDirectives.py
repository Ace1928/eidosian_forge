from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import uuid
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import matrix_creator_common
from googlecloudsdk.api_lib.firebase.test import matrix_ops
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
import six
def _BuildRoboDirectives(self, robo_directives_dict):
    """Build a list of RoboDirectives from the dictionary input."""
    robo_directives = []
    action_types = self._messages.RoboDirective.ActionTypeValueValuesEnum
    action_type_mapping = {'click': action_types.SINGLE_CLICK, 'text': action_types.ENTER_TEXT, 'ignore': action_types.IGNORE}
    for key, value in six.iteritems(robo_directives_dict or {}):
        action_type, resource_name = util.ParseRoboDirectiveKey(key)
        robo_directives.append(self._messages.RoboDirective(resourceName=resource_name, inputText=value, actionType=action_type_mapping.get(action_type)))
    return robo_directives