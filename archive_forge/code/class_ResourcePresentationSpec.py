from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.util.concepts import info_holders
class ResourcePresentationSpec(PresentationSpec):
    """Class that specifies how resource arguments are presented in a command."""

    def _ValidateFlagNameOverrides(self, flag_name_overrides):
        if not flag_name_overrides:
            return
        for attribute_name in flag_name_overrides.keys():
            for attribute in self.concept_spec.attributes:
                if attribute.name == attribute_name:
                    break
            else:
                raise ValueError('Attempting to override the name for an attribute not present in the concept: [{}]. Available attributes: [{}]'.format(attribute_name, ', '.join([attribute.name for attribute in self.concept_spec.attributes])))

    def _GetAttributeToArgsMap(self, flag_name_overrides):
        self._ValidateFlagNameOverrides(flag_name_overrides)
        attribute_to_args_map = {}
        for i, attribute in enumerate(self._concept_spec.attributes):
            is_anchor = i == len(self._concept_spec.attributes) - 1
            name = self.GetFlagName(attribute.name, self.name, flag_name_overrides, self.prefixes, is_anchor=is_anchor)
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
      is_anchor: bool, True if this it he anchor flag, False otherwise.

    Returns:
      (str) the name of the flag.
    """
        flag_name_overrides = flag_name_overrides or {}
        if attribute_name in flag_name_overrides:
            return flag_name_overrides.get(attribute_name)
        if attribute_name == 'project':
            return ''
        if is_anchor:
            return presentation_name
        prefix = util.PREFIX
        if prefixes:
            if presentation_name.startswith(util.PREFIX):
                prefix += presentation_name[len(util.PREFIX):] + '-'
            else:
                prefix += presentation_name.lower().replace('_', '-') + '-'
        return prefix + attribute_name

    def _GenerateInfo(self, fallthroughs_map):
        """Gets the ResourceInfo object for the ConceptParser.

    Args:
      fallthroughs_map: {str: [googlecloudsdk.calliope.concepts.deps.
        _FallthroughBase]}, dict keyed by attribute name to lists of
        fallthroughs.

    Returns:
      info_holders.ResourceInfo, the ResourceInfo object.
    """
        return info_holders.ResourceInfo(self.name, self.concept_spec, self.group_help, self.attribute_to_args_map, fallthroughs_map, required=self.required, plural=self.plural, group=self.group, hidden=self.hidden)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.name == other.name and self.concept_spec == other.concept_spec and (self.group_help == other.group_help) and (self.prefixes == other.prefixes) and (self.plural == other.plural) and (self.required == other.required) and (self.group == other.group) and (self.hidden == other.hidden)