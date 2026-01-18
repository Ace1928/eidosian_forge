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
def _get_event_wql_query(self, cls, event_type, timeframe=2, **where):
    """Return a WQL query used for polling WMI events.

            :param cls: the Hyper-V class polled for events.
            :param event_type: the type of event expected.
            :param timeframe: check for events that occurred in
                              the specified timeframe.
            :param where: key-value arguments which are to be included in the
                          query. For example: like=dict(foo="bar").
        """
    like = where.pop('like', {})
    like_str = ' AND '.join(("TargetInstance.%s LIKE '%s%%'" % (k, v) for k, v in like.items()))
    like_str = 'AND ' + like_str if like_str else ''
    query = "SELECT * FROM %(event_type)s WITHIN %(timeframe)s WHERE TargetInstance ISA '%(class)s' %(like)s" % {'class': cls, 'event_type': event_type, 'like': like_str, 'timeframe': timeframe}
    return query