from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.util.concepts import completers
from googlecloudsdk.core.util import text
import six
from six.moves import filter  # pylint: disable=redefined-builtin
def GetGroupHelp(self):
    base_text = super(MultitypeResourceInfo, self).GetGroupHelp()
    all_types = [type_.name for type_ in self.concept_spec.type_enum]
    return base_text + ' This resource can be one of the following types: [{}].'.format(', '.join(all_types))