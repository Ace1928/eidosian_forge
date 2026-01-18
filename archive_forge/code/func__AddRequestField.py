from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.security_policies import client
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.security_policies import flags as security_policy_flags
from googlecloudsdk.command_lib.compute.security_policies.rules import flags
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
@classmethod
def _AddRequestField(cls, compute_client, existing_request_fields, request_field_to_add):
    """Adds Request Field."""
    new_request_field = cls._ConvertRequestFieldToAdd(compute_client, request_field_to_add)
    for existing_request_field in existing_request_fields:
        if existing_request_field == new_request_field:
            return
    existing_request_fields.append(new_request_field)