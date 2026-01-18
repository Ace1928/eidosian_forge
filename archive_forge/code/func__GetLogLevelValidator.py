from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.service_extensions import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
def _GetLogLevelValidator():
    return arg_parsers.CustomFunctionValidator(lambda k: k in LOG_LEVEL_VALUES, 'Only the following keys are valid for log level: [{}].'.format(', '.join(LOG_LEVEL_VALUES)), lambda x: x.upper())