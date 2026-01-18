import functools
import inspect
import logging
from oslo_config import cfg
from oslo_log._i18n import _
def _get_safe_to_remove_release(release, remove_in):
    if remove_in is None:
        remove_in = 0
    new_release = chr(ord(release) + remove_in)
    if new_release in _RELEASES:
        return _RELEASES[new_release]
    else:
        return new_release