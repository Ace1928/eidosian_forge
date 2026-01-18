from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import os
import re
import textwrap
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.resource import yaml_printer
from googlecloudsdk.core.util import files
import six
def _AddRequiredFields(spec, fields):
    """Adds required fields to spec."""
    required = _GetRequiredFields(fields)
    if required:
        spec['required'] = sorted(required)