from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.util.concepts import info_holders
def _GetAttributeToArgsMap(self, flag_name_overrides):
    attribute_to_args_map = {}
    leaf_anchors = [a for a in self._concept_spec.attributes if self._concept_spec.IsLeafAnchor(a)]
    for attribute in self._concept_spec.attributes:
        is_anchor = [attribute] == leaf_anchors
        name = self.GetFlagName(attribute.name, self.name, flag_name_overrides=flag_name_overrides, prefixes=self.prefixes, is_anchor=is_anchor)
        if name:
            attribute_to_args_map[attribute.name] = name
    return attribute_to_args_map