from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_filter
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import peek_iterable
Replaces resource with its URI or skips the resource if it has no URI.

    Args:
      resource: The resource to replace with its URI.

    Returns:
      TapInjector(URI, replace=True) if the resource has a URI or False to skip
      the resource.
    