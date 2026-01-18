from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import re
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import util as format_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import http_encoding
import six
def _ParseCustomMappingsFromTuples(self):
    """Parses choice to enum mappings from custom_mapping with tuples.

     Parses choice mappings from dict mapping Enum strings to a tuple of
     choice values and choice help {str -> (str, str)} mapping.

    Raises:
      TypeError - Custom choices are not not valid (str,str) tuples.
    """
    self._choice_to_enum = {}
    self._enum_to_choice = {}
    self._choices = collections.OrderedDict()
    for enum_string, (choice, help_str) in sorted(six.iteritems(self._custom_mappings)):
        self._choice_to_enum[choice] = self._enum(enum_string)
        self._enum_to_choice[enum_string] = choice
        self._choices[choice] = help_str