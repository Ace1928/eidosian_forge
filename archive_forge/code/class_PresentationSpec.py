from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.util.concepts import info_holders
class PresentationSpec(object):
    """Class that defines how concept arguments are presented in a command.

  Attributes:
    name: str, the name of the main arg for the concept. Can be positional or
      flag style (UPPER_SNAKE_CASE or --lower-train-case).
    concept_spec: googlecloudsdk.calliope.concepts.ConceptSpec, The spec that
      specifies the concept.
    group_help: str, the help text for the entire arg group.
    prefixes: bool, whether to use prefixes before the attribute flags, such as
      `--myresource-project`.
    required: bool, whether the anchor argument should be required. If True, the
      command will fail at argparse time if the anchor argument isn't given.
    plural: bool, True if the resource will be parsed as a list, False
      otherwise.
    group: the parser or subparser for a Calliope command that the resource
      arguments should be added to. If not provided, will be added to the main
      parser.
    attribute_to_args_map: {str: str}, dict of attribute names to names of
      associated arguments.
    hidden: bool, True if the arguments should be hidden.
  """

    def __init__(self, name, concept_spec, group_help, prefixes=False, required=False, flag_name_overrides=None, plural=False, group=None, hidden=False):
        """Initializes a ResourcePresentationSpec.

    Args:
      name: str, the name of the main arg for the concept.
      concept_spec: googlecloudsdk.calliope.concepts.ConceptSpec, The spec that
        specifies the concept.
      group_help: str, the help text for the entire arg group.
      prefixes: bool, whether to use prefixes before the attribute flags, such
        as `--myresource-project`. This will match the "name" (in flag format).
      required: bool, whether the anchor argument should be required.
      flag_name_overrides: {str: str}, dict of attribute names to the desired
        flag name. To remove a flag altogether, use '' as its rename value.
      plural: bool, True if the resource will be parsed as a list, False
        otherwise.
      group: the parser or subparser for a Calliope command that the resource
        arguments should be added to. If not provided, will be added to the main
        parser.
      hidden: bool, True if the arguments should be hidden.
    """
        self.name = name
        self._concept_spec = concept_spec
        self.group_help = group_help
        self.prefixes = prefixes
        self.required = required
        self.plural = plural
        self.group = group
        self._attribute_to_args_map = self._GetAttributeToArgsMap(flag_name_overrides)
        self.hidden = hidden

    @property
    def concept_spec(self):
        """The ConceptSpec associated with the PresentationSpec.

    Returns:
      (googlecloudsdk.calliope.concepts.ConceptSpec) the concept spec.
    """
        return self._concept_spec

    @property
    def attribute_to_args_map(self):
        """The map of attribute names to associated args.

    Returns:
      {str: str}, the map.
    """
        return self._attribute_to_args_map

    def _GenerateInfo(self, fallthroughs_map):
        """Generate a ConceptInfo object for the ConceptParser.

    Must be overridden in subclasses.

    Args:
      fallthroughs_map: {str: [googlecloudsdk.calliope.concepts.deps.
        _FallthroughBase]}, dict keyed by attribute name to lists of
        fallthroughs.

    Returns:
      info_holders.ConceptInfo, the ConceptInfo object.
    """
        raise NotImplementedError

    def _GetAttributeToArgsMap(self, flag_name_overrides):
        """Generate a map of attributes to primary arg names.

    Must be overridden in subclasses.

    Args:
      flag_name_overrides: {str: str}, the dict of flags to overridden names.

    Returns:
      {str: str}, dict from attribute names to arg names.
    """
        raise NotImplementedError