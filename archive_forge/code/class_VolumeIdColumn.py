import copy
import functools
import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class VolumeIdColumn(cliff_columns.FormattableColumn):
    """Formattable column for volume ID column.

    Unlike the parent FormattableColumn class, the initializer of the
    class takes volume_cache as the second argument.
    osc_lib.utils.get_item_properties instantiate cliff FormattableColumn
    object with a single parameter "column value", so you need to pass
    a partially initialized class like
    ``functools.partial(VolumeIdColumn, volume_cache)``.
    """

    def __init__(self, value, volume_cache=None):
        super(VolumeIdColumn, self).__init__(value)
        self._volume_cache = volume_cache or {}

    def human_readable(self):
        """Return a volume name if available

        :rtype: either the volume ID or name
        """
        volume_id = self._value
        volume = volume_id
        if volume_id in self._volume_cache.keys():
            volume = self._volume_cache[volume_id].name
        return volume