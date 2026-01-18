import functools
import re
from eventlet import patcher
from eventlet import tpool
from oslo_log import log as logging
from oslo_utils import units
import six
from os_win._i18n import _
from os_win import conf
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
from os_win.utils import jobutils
def _prepare_profile_sd(self, **kwargs):
    profile_id_settings = self._create_default_setting_data(self._PORT_PROFILE_SET_DATA)
    for argument_name, attr_name in _PORT_PROFILE_ATTR_MAP.items():
        attribute = kwargs.pop(argument_name, None)
        if attribute is None:
            continue
        setattr(profile_id_settings, attr_name, attribute)
    if kwargs:
        raise TypeError('Unrecognized attributes %r' % kwargs)
    return profile_id_settings