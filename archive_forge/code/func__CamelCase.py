from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import logging
import os
from apitools.gen import gen_client
from googlecloudsdk.api_lib.regen import api_def
from googlecloudsdk.api_lib.regen import resource_generator
from googlecloudsdk.core.util import files
from mako import runtime
from mako import template
import six
def _CamelCase(snake_case):
    return ''.join((x.capitalize() for x in snake_case.split('_')))