from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import code
import site  # pylint: disable=unused-import
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.generated_clients.apis import apis_map
from googlecloudsdk.core import log  # pylint: disable=unused-import
from googlecloudsdk.core import properties  # pylint: disable=unused-import
from googlecloudsdk.core.console import console_io  # pylint: disable=unused-import
from googlecloudsdk.core.util import files  # pylint: disable=unused-import
def _PopulateApiNamesWithLoadMessage():
    """Make API names print instructions for loading the APIs when __repr__'ed.

  For example:

  >>> appengine
  Run `LoadApis()` to load all APIs, including this one.

  Load APIs it lazily because it takes about a second to load all APIs.
  """
    load_apis_message = 'Run `{0}()` to load all APIs, including this one.'.format(LoadApis.__name__)

    class _LoadApisMessage(object):

        def __repr__(self):
            return load_apis_message
    for api_name in apis_map.MAP:
        globals()[api_name] = _LoadApisMessage()