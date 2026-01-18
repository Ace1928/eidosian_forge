import datetime
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resource
from oslo_log import log as logging
from oslo_utils import timeutils
Utility class to encapsulate Cooldown related logic.

    This logic includes both cooldown timestamp comparing and scaling in
    progress checking.
    