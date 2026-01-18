from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
import six
def _IsLower(c):
    """Returns True if c is lower case or a caseless ideograph."""
    return c.isalpha() and (c.islower() or not c.isupper())