from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import display_taps
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import module_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.cache import cache_update_ops
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.resource import resource_reference
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.util import peek_iterable
import six
def _AddDisplayTaps(self):
    """Adds each of the standard display taps, if needed.

       The taps must be included in this order in order to generate the correct
       results. For example, limiting should not happen until after filtering is
       complete, and pagination should only happen on the fully trimmed results.
    """
    self._AddUriCacheTap()
    self._AddFlattenTap()
    self._AddFilterTap()
    self._AddSortByTap()
    self._AddLimitTap()
    self._AddPageTap()
    self._AddUriReplaceTap()