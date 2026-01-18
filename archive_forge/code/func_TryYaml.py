from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.endpoints import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import retry
import six
def TryYaml():
    try:
        return yaml.load(input_string)
    except yaml.YAMLParseError as e:
        if hasattr(e.inner_error, 'problem_mark'):
            mark = e.inner_error.problem_mark
            log.error('Service config YAML had an error at position (%s:%s)' % (mark.line + 1, mark.column + 1))