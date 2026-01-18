from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.tasks import app
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import http_encoding
import six  # pylint: disable=unused-import
from six.moves import filter  # pylint:disable=redefined-builtin
from six.moves import map  # pylint:disable=redefined-builtin
def ParseTask(task, queue_ref=None):
    """Parses an id or uri for a task."""
    params = queue_ref.AsDict() if queue_ref else None
    try:
        return resources.REGISTRY.Parse(task, collection=constants.TASKS_COLLECTION, params=params)
    except resources.RequiredFieldOmittedException:
        raise FullTaskUnspecifiedError('Must specify either the fully qualified task path or the queue flag.')