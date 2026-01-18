from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from collections import OrderedDict
import contextlib
import copy
import datetime
import json
import logging
import os
import sys
import time
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console.style import parser as style_parser
from googlecloudsdk.core.console.style import text
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def CreatedResource(resource, kind=None, is_async=False, details=None, failed=None):
    """Prints a status message indicating that a resource was created.

  Args:
    resource: str, The resource name.
    kind: str, The resource kind (instance, cluster, project, etc.).
    is_async: bool, True if the operation is in progress.
    details: str, Extra details appended to the message. Keep it succinct.
    failed: str, Failure message.
  """
    _PrintResourceChange('create', resource, kind, is_async, details, failed)