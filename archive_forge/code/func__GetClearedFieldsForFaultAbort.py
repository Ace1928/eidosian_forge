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
def _GetClearedFieldsForFaultAbort(fault_abort, field_prefix):
    """Gets a list of fields cleared by the user for HttpFaultAbort."""
    cleared_fields = []
    if not fault_abort.httpStatus:
        cleared_fields.append(field_prefix + 'httpStatus')
    if not fault_abort.percentage:
        cleared_fields.append(field_prefix + 'percentage')
    return cleared_fields