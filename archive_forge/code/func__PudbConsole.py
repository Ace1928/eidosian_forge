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
def _PudbConsole():
    """Run a console based on PuDB."""
    try:
        import pudb
        pudb.set_trace()
    except ImportError:
        log.error('Could not start the PuDB debugger. Please ensure that it is installed, or try the default debugger with `--mode=python`.')