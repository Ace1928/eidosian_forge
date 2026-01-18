from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.concepts import base
from googlecloudsdk.command_lib.concepts import dependency_managers
from googlecloudsdk.command_lib.concepts import names
import six
class RuntimeParser(object):
    """An object to manage parsing all concepts via their attributes.

  Once argument parsing is complete and ParseConcepts is called, each parsed
  concept is stored on this runtime parser as an attribute, named after the
  name of that concept.

  Attributes:
    parsed_args: the argparse namespace after arguments have been parsed.
    <CONCEPT_NAME> (the namespace format of each top level concept, such as
      "foo_bar"): the parsed concept corresponding to that name.
  """

    def __init__(self, attributes):
        self.parsed_args = None
        self._attributes = {}
        for attribute in attributes:
            attr_name = names.ConvertToNamespaceName(attribute.concept.GetPresentationName())
            if attr_name in self._attributes:
                raise ValueError('Attempting to add two concepts with the same presentation name: [{}]'.format(attr_name))
            self._attributes[attr_name] = attribute

    def ParseConcepts(self):
        """Parse all concepts.

    Stores the result of parsing concepts, keyed to the namespace format of
    their presentation name. Afterward, will be accessible as
    args.<LOWER_SNAKE_CASE_NAME>.

    Raises:
      googlecloudsdk.command_lib.concepts.exceptions.Error: if parsing fails.
    """
        final = {}
        for attr_name, attribute in six.iteritems(self._attributes):
            dependencies = dependency_managers.DependencyNode.FromAttribute(attribute)
            final[attr_name] = FinalParse(dependencies, self.ParsedArgs)
        for name, value in six.iteritems(final):
            setattr(self.parsed_args, name, value)

    def ParsedArgs(self):
        """A lazy property to use during concept parsing.

    Returns:
      googlecloudsdk.calliope.parser_extensions.Namespace: the parsed argparse
        namespace | None, if the parser hasn't been registered to the namespace
        yet.
    """
        return self.parsed_args