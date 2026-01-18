from __future__ import annotations
from typing import Any, Optional  # noqa: H301
from oslo_log import log as logging
from oslo_utils import netutils
@classmethod
def _get_rbd_args(cls, connection_properties: dict[str, Any], conf: Optional[str]=None) -> list[str]:
    user = connection_properties.get('auth_username')
    monitor_ips = connection_properties.get('hosts')
    monitor_ports = connection_properties.get('ports')
    args: list[str] = []
    if user:
        args = ['--id', user]
    if monitor_ips and monitor_ports:
        monitors = ['%s:%s' % (ip, port) for ip, port in zip(cls._sanitize_mon_hosts(monitor_ips), monitor_ports)]
        for monitor in monitors:
            args += ['--mon_host', monitor]
    if conf:
        args += ['--conf', conf]
    return args