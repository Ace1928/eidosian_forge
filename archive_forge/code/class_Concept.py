from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import six
class Concept(six.with_metaclass(abc.ABCMeta, object)):
    """Abstract base class for concept args.

  Attributes:
    name: str, the name of the concept. Used to determine
      the argument or group name of the concept.
    key: str, the name by which the parsed concept is stored in the dependency
      view. If not given, is the same as the concept's name. Generally,
      this should only be set and used by containing concepts when parsing
      from a DependencyView object. End users of concepts do not need to
      use it.
    help_text: str, the help text to be displayed for this concept.
    required: bool, whether the concept must be provided by the end user. If
      False, it's acceptable to have an empty result; otherwise, an empty
      result will raise an error.
    hidden: bool, whether the associated argument or group should be hidden.
  """

    def __init__(self, name, key=None, help_text='', required=False, hidden=False):
        self.name = name
        self.help_text = help_text
        self.required = required
        self.hidden = hidden
        self.key = key or self.name

    @abc.abstractmethod
    def Attribute(self):
        """Returns an Attribute object representing the attributes.

    Must be defined in subclasses.

    Returns:
      Attribute | AttributeGroup, the attribute or group.
    """
        raise NotImplementedError

    @abc.abstractmethod
    def GetPresentationName(self):
        """Returns the main name for the concept."""
        raise NotImplementedError

    def BuildHelpText(self):
        """Builds and returns the help text.

    Returns:
      str, the help text for the concept.
    """
        return self.help_text

    def Marshal(self):
        """Returns the list of concepts that this concept marshals."""
        return None

    @abc.abstractmethod
    def Parse(self, dependencies):
        """Parses the concept.

    Args:
      dependencies: a DependenciesView object.

    Returns:
      the parsed concept.

    Raises:
      googlecloudsdk.command_lib.concepts.exceptions.Error, if parsing fails.
    """
        raise NotImplementedError

    @abc.abstractmethod
    def IsArgRequired(self):
        """Returns whether this concept is required to be specified by argparse."""
        return False

    def MakeArgKwargs(self):
        """Returns argparse kwargs shared between all concept types."""
        return {'help': self.BuildHelpText(), 'required': self.IsArgRequired(), 'hidden': self.hidden}