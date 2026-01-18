from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import multitype
from googlecloudsdk.command_lib.secrets import completers as secrets_completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import resources
def MakeGetUriFunc(collection: str, api_version: str='v1'):
    """Returns a function which turns a resource into a uri.

  Example:
    class List(base.ListCommand):
      def GetUriFunc(self):
        return MakeGetUriFunc(self)

  Args:
    collection: A command instance.
    api_version: api_version to be displayed.

  Returns:
    A function which can be returned in GetUriFunc.
  """

    def _GetUri(resource):
        registry = resources.REGISTRY.Clone()
        registry.RegisterApiByName('secretmanager', api_version)
        parsed = registry.Parse(resource.name, collection=collection)
        return parsed.SelfLink()
    return _GetUri