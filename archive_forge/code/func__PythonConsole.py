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
def _PythonConsole():
    """Run a console based on the built-in code.InteractiveConsole."""
    try:
        import readline
        import rlcompleter
    except ImportError:
        pass
    else:
        readline.set_completer(rlcompleter.Completer(globals()).complete)
        readline.parse_and_bind('tab: complete')
    console = code.InteractiveConsole(globals())
    console.interact(_BANNER)