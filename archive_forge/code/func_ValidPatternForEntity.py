from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def ValidPatternForEntity(entity_name):
    """Returns a compiled regex that matches valid values for `entity_name`."""
    return re.compile(_ValidPatternForEntity(entity_name))