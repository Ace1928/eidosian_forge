from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import copy
import io
import json
import textwrap
from apitools.base.py import encoding
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_diff
from googlecloudsdk.core.util import edit
import six
def GetCurrentAuthorizedView(authorized_view_name, check_ascii):
    """Get the current authorized view resource object given the authorized view name.

  Args:
    authorized_view_name: The name of the authorized view.
    check_ascii: True if we should check to make sure that the returned
      authorized view contains only ascii characters.

  Returns:
    The view resource object.

  Raises:
    ValueError if check_ascii is true and the current authorized view definition
    contains invalid non-ascii characters.
  """
    client = util.GetAdminClient()
    request = util.GetAdminMessages().BigtableadminProjectsInstancesTablesAuthorizedViewsGetRequest(name=authorized_view_name)
    try:
        authorized_view = client.projects_instances_tables_authorizedViews.Get(request)
        if check_ascii:
            CheckOnlyAsciiCharactersInAuthorizedView(authorized_view)
        return authorized_view
    except api_exceptions.HttpError as error:
        raise exceptions.HttpException(error)