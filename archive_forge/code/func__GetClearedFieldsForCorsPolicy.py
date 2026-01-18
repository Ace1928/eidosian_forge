from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.url_maps import flags
from googlecloudsdk.command_lib.compute.url_maps import url_maps_utils
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.console import console_io
def _GetClearedFieldsForCorsPolicy(cors_policy, field_prefix):
    """Gets a list of fields cleared by the user for CorsPolicy."""
    cleared_fields = []
    if not cors_policy.allowOrigins:
        cleared_fields.append(field_prefix + 'allowOrigins')
    if not cors_policy.allowOriginRegexes:
        cleared_fields.append(field_prefix + 'allowOriginRegexes')
    if not cors_policy.allowMethods:
        cleared_fields.append(field_prefix + 'allowMethods')
    if not cors_policy.allowHeaders:
        cleared_fields.append(field_prefix + 'allowHeaders')
    if not cors_policy.exposeHeaders:
        cleared_fields.append(field_prefix + 'exposeHeaders')
    if not cors_policy.maxAge:
        cleared_fields.append(field_prefix + 'maxAge')
    if not cors_policy.allowCredentials:
        cleared_fields.append(field_prefix + 'allowCredentials')
    if not cors_policy.disabled:
        cleared_fields.append(field_prefix + 'disabled')
    return cleared_fields