import re
import ssl
import urllib.parse
import dogpile.cache
from dogpile.cache import api
from dogpile.cache import proxy
from dogpile.cache import util
from oslo_log import log
from oslo_utils import importutils
from oslo_cache._i18n import _
from oslo_cache import _opts
from oslo_cache import exception
def _get_expiration_time_fn(conf, group):
    """Build a function that returns a config group's expiration time status.

    For any given object that has caching capabilities, an int config option
    called ``cache_time`` for that driver's group should exist and typically
    default to ``None``. This function will use that value to tell the caching
    decorator of the TTL override for caching the resulting objects. If the
    value of the config option is ``None`` the default value provided in the
    ``[cache] expiration_time`` option will be used by the decorator. The
    default may be set to something other than ``None`` in cases where the
    caching TTL should not be tied to the global default(s).

    To properly use this with the decorator, pass this function the
    configuration group and assign the result to a variable. Pass the new
    variable to the caching decorator as the named argument
    ``expiration_time``.

    :param group: name of the configuration group to examine
    :type group: string
    :rtype: function reference
    """

    def get_expiration_time():
        conf_group = getattr(conf, group)
        return getattr(conf_group, 'cache_time', None)
    return get_expiration_time