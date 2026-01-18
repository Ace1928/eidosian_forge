from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import re
from apitools.base.protorpclite import messages as proto_messages
from apitools.base.py import encoding as apitools_encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import files
import six
def RemoveTriggerSubstitutions(old_substitutions, substitutions_to_be_removed, messages):
    """Removes existing substitutions for the update command.

  Args:
    old_substitutions: The existing substitutions.
    substitutions_to_be_removed: The substitutions to be removed if exist.
    messages: A Cloud Build messages module.

  Returns:
    The updated trigger substitutions.
  """
    if not substitutions_to_be_removed:
        return None
    substitution_properties = []
    if old_substitutions:
        for sub in old_substitutions.additionalProperties:
            if sub.key not in substitutions_to_be_removed:
                substitution_properties.append(messages.BuildTrigger.SubstitutionsValue.AdditionalProperty(key=sub.key, value=sub.value))
    if not substitution_properties:
        substitution_properties.append(messages.BuildTrigger.SubstitutionsValue.AdditionalProperty())
    return messages.BuildTrigger.SubstitutionsValue(additionalProperties=substitution_properties)