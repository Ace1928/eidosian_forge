from googlecloudsdk.command_lib.concepts import concept_managers
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import io
import re
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.command_lib.concepts import base
from googlecloudsdk.command_lib.concepts import dependency_managers
from googlecloudsdk.command_lib.concepts import exceptions
from googlecloudsdk.command_lib.concepts import names
from googlecloudsdk.core.util import scaled_integer
from googlecloudsdk.core.util import semver
from googlecloudsdk.core.util import times
import six
class ConceptType(SimpleArg):
    """Concept type base class.

  All concept types derive from this class. The methods implement lexing,
  parsing, constraints, help text, and formatting.
  """

    def Convert(self, string):
        """Converts a value from string and returns it.

    The converter must do syntax checking and raise actionable exceptions. All
    non-space characters in string must be consumed. This method may raise
    syntax exceptions, but otherwise does no validation.

    Args:
        string: The string to convert to a concept type value.

    Returns:
      The converted value.
    """
        return string

    def Display(self, value):
        """Returns the string representation of a parsed concept value.

    This method is the inverse of Convert() and Parse(). It returns the
    string representation of a parsed concept value that can be used in
    formatted messages.

    Args:
        value: The concept value to display.

    Returns:
      The string representation of a parsed concept value.
    """
        return six.text_type(value)

    def Normalize(self, value):
        """Returns the normalized value.

    Called after the value has been validated. It normalizes internal values
    for compatibility with other interfaces. This can be accomplished by
    subclassing with a shim class that contains only a Normalize() method.

    Args:
        value: The concept value to normalize.

    Returns:
      The normalized value.
    """
        return value

    def Parse(self, dependencies):
        """Converts, validates and normalizes a value string from dependencies."""
        string = super(ConceptType, self).Parse(dependencies)
        value = self.Convert(string)
        self.Validate(value)
        return self.Normalize(value)

    def Validate(self, value):
        """Validates value.

    Syntax checking has already been done by Convert(). The validator imposes
    additional constraints on valid values for the concept type and must raise
    actionable exceptions when the constraints are not met.

    Args:
      value: The concept value to validate.
    """
        pass