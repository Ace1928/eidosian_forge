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
def _GetClearedFieldsForUrlRedirect(url_redirect, field_prefix):
    """Gets a list of fields cleared by the user for UrlRedirect."""
    cleared_fields = []
    if not url_redirect.hostRedirect:
        cleared_fields.append(field_prefix + 'hostRedirect')
    if not url_redirect.pathRedirect:
        cleared_fields.append(field_prefix + 'pathRedirect')
    if not url_redirect.prefixRedirect:
        cleared_fields.append(field_prefix + 'prefixRedirect')
    if not url_redirect.redirectResponseCode:
        cleared_fields.append(field_prefix + 'redirectResponseCode')
    if not url_redirect.httpsRedirect:
        cleared_fields.append(field_prefix + 'httpsRedirect')
    if not url_redirect.stripQuery:
        cleared_fields.append(field_prefix + 'stripQuery')
    return cleared_fields