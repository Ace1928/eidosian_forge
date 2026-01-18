from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.util.concepts import info_holders
class MultitypeResourcePresentationSpec(PresentationSpec):
    """A resource-specific presentation spec."""

    def _GetAttributeToArgsMap(self, flag_name_overrides):
        attribute_to_args_map = {}
        leaf_anchors = [a for a in self._concept_spec.attributes if self._concept_spec.IsLeafAnchor(a)]
        for attribute in self._concept_spec.attributes:
            is_anchor = [attribute] == leaf_anchors
            name = self.GetFlagName(attribute.name, self.name, flag_name_overrides=flag_name_overrides, prefixes=self.prefixes, is_anchor=is_anchor)
            if name:
                attribute_to_args_map[attribute.name] = name
        return attribute_to_args_map

    @staticmethod
    def GetFlagName(attribute_name, presentation_name, flag_name_overrides=None, prefixes=False, is_anchor=False):
        """Gets the flag name for a given attribute name.

    Returns a flag name for an attribute, adding prefixes as necessary or using
    overrides if an override map is provided.

    Args:
      attribute_name: str, the name of the attribute to base the flag name on.
      presentation_name: str, the anchor argument name of the resource the
        attribute belongs to (e.g. '--foo').
      flag_name_overrides: {str: str}, a dict of attribute names to exact string
        of the flag name to use for the attribute. None if no overrides.
      prefixes: bool, whether to use the resource name as a prefix for the flag.
      is_anchor: bool, True if this is the anchor flag, False otherwise.

    Returns:
      (str) the name of the flag.
    """
        flag_name_overrides = flag_name_overrides or {}
        if attribute_name in flag_name_overrides:
            return flag_name_overrides.get(attribute_name)
        if is_anchor:
            return presentation_name
        if attribute_name == 'project':
            return ''
        if prefixes:
            return util.FlagNameFormat('-'.join([presentation_name, attribute_name]))
        return util.FlagNameFormat(attribute_name)

    def _GenerateInfo(self, fallthroughs_map):
        """Gets the MultitypeResourceInfo object for the ConceptParser.

    Args:
      fallthroughs_map: {str: [googlecloudsdk.calliope.concepts.deps.
        _FallthroughBase]}, dict keyed by attribute name to lists of
        fallthroughs.

    Returns:
      info_holders.MultitypeResourceInfo, the ResourceInfo object.
    """
        return info_holders.MultitypeResourceInfo(self.name, self.concept_spec, self.group_help, self.attribute_to_args_map, fallthroughs_map, required=self.required, plural=self.plural, group=self.group)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.name == other.name and self.concept_spec == other.concept_spec and (self.group_help == other.group_help) and (self.prefixes == other.prefixes) and (self.plural == other.plural) and (self.required == other.required) and (self.group == other.group) and (self.hidden == other.hidden)