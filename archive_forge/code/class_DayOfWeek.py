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
class DayOfWeek(TypeWithRegexConstraint):
    """Day of the week concept."""
    _DAYS = ['SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT']

    def BuildHelpText(self):
        """Appends day of week syntax to the original help text."""
        return "{}Must be a string representing a day of the week in English, such as 'MON' or 'FRI'. Case is ignored, and any characters after the first three are ignored.{}".format(_Insert(super(DayOfWeek, self).BuildHelpText()), _Append(self.Constraints()))

    def Convert(self, string):
        """Converts a day of week from string returns it."""
        if not string:
            return None
        value = string.upper()[:3]
        if value not in self._DAYS:
            raise exceptions.ParseError(self.GetPresentationName(), 'A day of week value [{}] must be one of: [{}].'.format(string, ', '.join(self._DAYS)))
        return value