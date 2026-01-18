import logging
import numbers
from os_ken.lib import ip
from os_ken.services.protocols.bgp.utils.validation import is_valid_ipv4
from os_ken.services.protocols.bgp.utils.validation import is_valid_asn
from os_ken.services.protocols.bgp import rtconf
from os_ken.services.protocols.bgp.rtconf.base import BaseConf
from os_ken.services.protocols.bgp.rtconf.base import BaseConfListener
from os_ken.services.protocols.bgp.rtconf.base import compute_optional_conf
from os_ken.services.protocols.bgp.rtconf.base import ConfigTypeError
from os_ken.services.protocols.bgp.rtconf.base import ConfigValueError
from os_ken.services.protocols.bgp.rtconf.base import MissingRequiredConf
from os_ken.services.protocols.bgp.rtconf.base import validate
class CommonConf(BaseConf):
    """Encapsulates configurations applicable to all peer sessions.

    Currently if any of these configurations change, it is assumed that current
    active peer session will be bought down and restarted.
    """
    CONF_CHANGED_EVT = 1
    VALID_EVT = frozenset([CONF_CHANGED_EVT])
    REQUIRED_SETTINGS = frozenset([ROUTER_ID, LOCAL_AS])
    OPTIONAL_SETTINGS = frozenset([REFRESH_STALEPATH_TIME, REFRESH_MAX_EOR_TIME, LABEL_RANGE, BGP_SERVER_HOSTS, BGP_SERVER_PORT, TCP_CONN_TIMEOUT, BGP_CONN_RETRY_TIME, MAX_PATH_EXT_RTFILTER_ALL, ALLOW_LOCAL_AS_IN_COUNT, CLUSTER_ID, LOCAL_PREF])

    def __init__(self, **kwargs):
        super(CommonConf, self).__init__(**kwargs)

    def _init_opt_settings(self, **kwargs):
        super(CommonConf, self)._init_opt_settings(**kwargs)
        self._settings[ALLOW_LOCAL_AS_IN_COUNT] = compute_optional_conf(ALLOW_LOCAL_AS_IN_COUNT, 0, **kwargs)
        self._settings[LABEL_RANGE] = compute_optional_conf(LABEL_RANGE, DEFAULT_LABEL_RANGE, **kwargs)
        self._settings[REFRESH_STALEPATH_TIME] = compute_optional_conf(REFRESH_STALEPATH_TIME, DEFAULT_REFRESH_STALEPATH_TIME, **kwargs)
        self._settings[REFRESH_MAX_EOR_TIME] = compute_optional_conf(REFRESH_MAX_EOR_TIME, DEFAULT_REFRESH_MAX_EOR_TIME, **kwargs)
        self._settings[BGP_SERVER_HOSTS] = compute_optional_conf(BGP_SERVER_HOSTS, DEFAULT_BGP_SERVER_HOSTS, **kwargs)
        self._settings[BGP_SERVER_PORT] = compute_optional_conf(BGP_SERVER_PORT, DEFAULT_BGP_SERVER_PORT, **kwargs)
        self._settings[TCP_CONN_TIMEOUT] = compute_optional_conf(TCP_CONN_TIMEOUT, DEFAULT_TCP_CONN_TIMEOUT, **kwargs)
        self._settings[BGP_CONN_RETRY_TIME] = compute_optional_conf(BGP_CONN_RETRY_TIME, DEFAULT_BGP_CONN_RETRY_TIME, **kwargs)
        self._settings[MAX_PATH_EXT_RTFILTER_ALL] = compute_optional_conf(MAX_PATH_EXT_RTFILTER_ALL, DEFAULT_MAX_PATH_EXT_RTFILTER_ALL, **kwargs)
        self._settings[CLUSTER_ID] = compute_optional_conf(CLUSTER_ID, kwargs[ROUTER_ID], **kwargs)
        self._settings[LOCAL_PREF] = compute_optional_conf(LOCAL_PREF, DEFAULT_LOCAL_PREF, **kwargs)

    @property
    def local_as(self):
        return self._settings[LOCAL_AS]

    @property
    def router_id(self):
        return self._settings[ROUTER_ID]

    @property
    def cluster_id(self):
        return self._settings[CLUSTER_ID]

    @property
    def allow_local_as_in_count(self):
        return self._settings[ALLOW_LOCAL_AS_IN_COUNT]

    @property
    def bgp_conn_retry_time(self):
        return self._settings[BGP_CONN_RETRY_TIME]

    @property
    def tcp_conn_timeout(self):
        return self._settings[TCP_CONN_TIMEOUT]

    @property
    def refresh_stalepath_time(self):
        return self._settings[REFRESH_STALEPATH_TIME]

    @property
    def refresh_max_eor_time(self):
        return self._settings[REFRESH_MAX_EOR_TIME]

    @property
    def label_range(self):
        return self._settings[LABEL_RANGE]

    @property
    def bgp_server_hosts(self):
        return self._settings[BGP_SERVER_HOSTS]

    @property
    def bgp_server_port(self):
        return self._settings[BGP_SERVER_PORT]

    @property
    def max_path_ext_rtfilter_all(self):
        return self._settings[MAX_PATH_EXT_RTFILTER_ALL]

    @property
    def local_pref(self):
        return self._settings[LOCAL_PREF]

    @classmethod
    def get_opt_settings(self):
        self_confs = super(CommonConf, self).get_opt_settings()
        self_confs.update(CommonConf.OPTIONAL_SETTINGS)
        return self_confs

    @classmethod
    def get_req_settings(self):
        self_confs = super(CommonConf, self).get_req_settings()
        self_confs.update(CommonConf.REQUIRED_SETTINGS)
        return self_confs

    @classmethod
    def get_valid_evts(self):
        self_valid_evts = super(CommonConf, self).get_valid_evts()
        self_valid_evts.update(CommonConf.VALID_EVT)
        return self_valid_evts

    def update(self, **kwargs):
        """Updates global configuration settings with given values.

        First checks if given configuration values differ from current values.
        If any of the configuration values changed, generates a change event.
        Currently we generate change event for any configuration change.
        Note: This method is idempotent.
        """
        super(CommonConf, self).update(**kwargs)
        conf_changed = False
        for conf_name, conf_value in kwargs.items():
            rtconf.base.get_validator(conf_name)(conf_value)
            item1 = self._settings.get(conf_name, None)
            item2 = kwargs.get(conf_name, None)
            if item1 != item2:
                conf_changed = True
        if conf_changed:
            for conf_name, conf_value in kwargs.items():
                self._settings[conf_name] = conf_value
            self._notify_listeners(CommonConf.CONF_CHANGED_EVT, self)