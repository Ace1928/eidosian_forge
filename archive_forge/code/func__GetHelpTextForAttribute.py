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
def _GetHelpTextForAttribute(self, attribute):
    base_text = super(MultitypeResourceInfo, self)._GetHelpTextForAttribute(attribute)
    relevant_types = sorted([type_.name for type_ in self.concept_spec._attribute_to_types_map.get(attribute.name)])
    all_types = [type_.name for type_ in self.concept_spec.type_enum]
    if len(set(relevant_types)) == len(all_types):
        return base_text
    return base_text + ' Must be specified for resource of type {}.'.format(' or '.join(['[{}]'.format(t) for t in relevant_types]))