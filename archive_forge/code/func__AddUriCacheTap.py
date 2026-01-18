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
def _AddUriCacheTap(self):
    """Taps a resource Uri cache updater into self.resources if needed."""
    from googlecloudsdk.calliope import base
    if self._cache_updater == cache_update_ops.NoCacheUpdater:
        return
    if not self._cache_updater:
        if not isinstance(self._command, (base.CreateCommand, base.DeleteCommand, base.ListCommand, base.RestoreCommand)):
            return
        if 'AddCacheUpdater' in properties.VALUES.core.lint.Get():
            raise CommandNeedsAddCacheUpdater('`{}` needs a parser.display_info.AddCacheUpdater() call.'.format(' '.join(self._args._GetCommand().GetPath())))
        return
    if any([self._GetFlag(flag) for flag in self._CORRUPT_FLAGS]):
        return
    if isinstance(self._command, (base.CreateCommand, base.RestoreCommand)):
        cache_update_op = cache_update_ops.AddToCacheOp(self._cache_updater)
    elif isinstance(self._command, base.DeleteCommand):
        cache_update_op = cache_update_ops.DeleteFromCacheOp(self._cache_updater)
    elif isinstance(self._command, base.ListCommand):
        cache_update_op = cache_update_ops.ReplaceCacheOp(self._cache_updater)
    else:
        raise CommandShouldntHaveAddCacheUpdater('Cache updater [{}] not expected for [{}] `{}`.'.format(module_util.GetModulePath(self._cache_updater), module_util.GetModulePath(self._args._GetCommand()), ' '.join(self._args._GetCommand().GetPath())))
    tap = display_taps.UriCacher(cache_update_op, self._transform_uri)
    self._resources = peek_iterable.Tapper(self._resources, tap)