import abc
from abc import ABCMeta
from abc import abstractmethod
import functools
import numbers
import logging
import uuid
from os_ken.services.protocols.bgp.base import add_bgp_error_metadata
from os_ken.services.protocols.bgp.base import BGPSException
from os_ken.services.protocols.bgp.base import get_validator
from os_ken.services.protocols.bgp.base import RUNTIME_CONF_ERROR_CODE
from os_ken.services.protocols.bgp.base import validate
from os_ken.services.protocols.bgp.utils import validation
from os_ken.services.protocols.bgp.utils.validation import is_valid_asn
class ConfWithStats(BaseConf):
    """Configuration settings related to statistics collection."""
    STATS_LOG_ENABLED = 'statistics_log_enabled'
    DEFAULT_STATS_LOG_ENABLED = False
    STATS_TIME = 'statistics_interval'
    DEFAULT_STATS_TIME = 60
    UPDATE_STATS_LOG_ENABLED_EVT = 'update_stats_log_enabled_evt'
    UPDATE_STATS_TIME_EVT = 'update_stats_time_evt'
    VALID_EVT = frozenset([UPDATE_STATS_LOG_ENABLED_EVT, UPDATE_STATS_TIME_EVT])
    OPTIONAL_SETTINGS = frozenset([STATS_LOG_ENABLED, STATS_TIME])

    def __init__(self, **kwargs):
        super(ConfWithStats, self).__init__(**kwargs)

    def _init_opt_settings(self, **kwargs):
        super(ConfWithStats, self)._init_opt_settings(**kwargs)
        self._settings[ConfWithStats.STATS_LOG_ENABLED] = compute_optional_conf(ConfWithStats.STATS_LOG_ENABLED, ConfWithStats.DEFAULT_STATS_LOG_ENABLED, **kwargs)
        self._settings[ConfWithStats.STATS_TIME] = compute_optional_conf(ConfWithStats.STATS_TIME, ConfWithStats.DEFAULT_STATS_TIME, **kwargs)

    @property
    def stats_log_enabled(self):
        return self._settings[ConfWithStats.STATS_LOG_ENABLED]

    @stats_log_enabled.setter
    def stats_log_enabled(self, enabled):
        get_validator(ConfWithStats.STATS_LOG_ENABLED)(enabled)
        if enabled != self.stats_log_enabled:
            self._settings[ConfWithStats.STATS_LOG_ENABLED] = enabled
            self._notify_listeners(ConfWithStats.UPDATE_STATS_LOG_ENABLED_EVT, enabled)

    @property
    def stats_time(self):
        return self._settings[ConfWithStats.STATS_TIME]

    @stats_time.setter
    def stats_time(self, stats_time):
        get_validator(ConfWithStats.STATS_TIME)(stats_time)
        if stats_time != self.stats_time:
            self._settings[ConfWithStats.STATS_TIME] = stats_time
            self._notify_listeners(ConfWithStats.UPDATE_STATS_TIME_EVT, stats_time)

    @classmethod
    def get_opt_settings(cls):
        confs = super(ConfWithStats, cls).get_opt_settings()
        confs.update(ConfWithStats.OPTIONAL_SETTINGS)
        return confs

    @classmethod
    def get_valid_evts(cls):
        valid_evts = super(ConfWithStats, cls).get_valid_evts()
        valid_evts.update(ConfWithStats.VALID_EVT)
        return valid_evts

    def update(self, **kwargs):
        super(ConfWithStats, self).update(**kwargs)
        self.stats_log_enabled = compute_optional_conf(ConfWithStats.STATS_LOG_ENABLED, ConfWithStats.DEFAULT_STATS_LOG_ENABLED, **kwargs)
        self.stats_time = compute_optional_conf(ConfWithStats.STATS_TIME, ConfWithStats.DEFAULT_STATS_TIME, **kwargs)