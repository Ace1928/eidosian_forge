from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class _ProjectNumberPropertyFallthrough(deps.PropertyFallthrough):
    """A fallthrough for project number from property core/project."""

    def __init__(self):
        """See base class."""
        super(_ProjectNumberPropertyFallthrough, self).__init__(prop=properties.VALUES.core.project)

    def _Call(self, parsed_args):
        """See base class."""
        return _EnsureProjectNumber(super(_ProjectNumberPropertyFallthrough, self)._Call(parsed_args))