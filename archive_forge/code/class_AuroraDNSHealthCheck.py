import hmac
import json
import base64
import datetime
from hashlib import sha256
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, ProviderError, InvalidCredsError
class AuroraDNSHealthCheck:
    """
    AuroraDNS Healthcheck resource.
    """

    def __init__(self, id, type, hostname, ipaddress, port, interval, path, threshold, health, enabled, zone, driver, extra=None):
        """
        :param id: Healthcheck id
        :type id: ``str``

        :param hostname: Hostname or FQDN of the target
        :type hostname: ``str``

        :param ipaddress: IPv4 or IPv6 address of the target
        :type ipaddress: ``str``

        :param port: The port on the target to monitor
        :type port: ``int``

        :param interval: The interval of the health check
        :type interval: ``int``

        :param path: The path to monitor on the target
        :type path: ``str``

        :param threshold: The threshold of before marking a check as failed
        :type threshold: ``int``

        :param health: The current health of the health check
        :type health: ``bool``

        :param enabled: If the health check is currently enabled
        :type enabled: ``bool``

        :param zone: Zone instance.
        :type zone: :class:`Zone`

        :param driver: DNSDriver instance.
        :type driver: :class:`DNSDriver`

        :param extra: (optional) Extra attributes (driver specific).
        :type extra: ``dict``
        """
        self.id = str(id) if id else None
        self.type = type
        self.hostname = hostname
        self.ipaddress = ipaddress
        self.port = int(port) if port else None
        self.interval = int(interval)
        self.path = path
        self.threshold = int(threshold)
        self.health = bool(health)
        self.enabled = bool(enabled)
        self.zone = zone
        self.driver = driver
        self.extra = extra or {}

    def update(self, type=None, hostname=None, ipaddress=None, port=None, interval=None, path=None, threshold=None, enabled=None, extra=None):
        return self.driver.ex_update_healthcheck(healthcheck=self, type=type, hostname=hostname, ipaddress=ipaddress, port=port, path=path, interval=interval, threshold=threshold, enabled=enabled, extra=extra)

    def delete(self):
        return self.driver.ex_delete_healthcheck(healthcheck=self)

    def __repr__(self):
        return '<AuroraDNSHealthCheck: zone=%s, id=%s, type=%s, hostname=%s, ipaddress=%s, port=%d, interval=%d, health=%s, provider=%s...>' % (self.zone.id, self.id, self.type, self.hostname, self.ipaddress, self.port, self.interval, self.health, self.driver.name)