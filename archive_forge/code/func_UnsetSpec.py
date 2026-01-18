from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.accesscontextmanager import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import log
from googlecloudsdk.core import resources as core_resources
import six
def UnsetSpec(self, perimeter_ref, use_explicit_dry_run_spec):
    """Unsets the spec for a Service Perimeter.

    Args:
      perimeter_ref: resources.Resource, reference to the perimeter to patch.
      use_explicit_dry_run_spec: The value to use for the perimeter field of the
        same name.

    Returns:
      ServicePerimeter, the updated Service Perimeter.
    """
    perimeter = self.messages.ServicePerimeter()
    perimeter.useExplicitDryRunSpec = use_explicit_dry_run_spec
    perimeter.spec = None
    update_mask = ['spec', 'useExplicitDryRunSpec']
    return self._ApplyPatch(perimeter_ref, perimeter, update_mask)