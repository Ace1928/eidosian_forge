from __future__ import absolute_import
import sys
import time
from sentry_sdk.api import continue_trace
from sentry_sdk.consts import OP
from sentry_sdk._compat import reraise
from sentry_sdk._functools import wraps
from sentry_sdk.crons import capture_checkin, MonitorStatus
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.tracing import BAGGAGE_HEADER_NAME, TRANSACTION_SOURCE_TASK
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import (
def _get_monitor_config(celery_schedule, app, monitor_name):
    monitor_config = {}
    schedule_type = None
    schedule_value = None
    schedule_unit = None
    if isinstance(celery_schedule, crontab):
        schedule_type = 'crontab'
        schedule_value = '{0._orig_minute} {0._orig_hour} {0._orig_day_of_month} {0._orig_month_of_year} {0._orig_day_of_week}'.format(celery_schedule)
    elif isinstance(celery_schedule, schedule):
        schedule_type = 'interval'
        schedule_value, schedule_unit = _get_humanized_interval(celery_schedule.seconds)
        if schedule_unit == 'second':
            logger.warning("Intervals shorter than one minute are not supported by Sentry Crons. Monitor '%s' has an interval of %s seconds. Use the `exclude_beat_tasks` option in the celery integration to exclude it.", monitor_name, schedule_value)
            return {}
    else:
        logger.warning("Celery schedule type '%s' not supported by Sentry Crons.", type(celery_schedule))
        return {}
    monitor_config['schedule'] = {}
    monitor_config['schedule']['type'] = schedule_type
    monitor_config['schedule']['value'] = schedule_value
    if schedule_unit is not None:
        monitor_config['schedule']['unit'] = schedule_unit
    monitor_config['timezone'] = hasattr(celery_schedule, 'tz') and celery_schedule.tz is not None and str(celery_schedule.tz) or app.timezone or 'UTC'
    return monitor_config