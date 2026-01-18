from oslo_log import log as logging
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
def _get_zone(self, zone_name, ignore_missing=True):
    zones = self._dns_manager.MicrosoftDNS_Zone(Name=zone_name)
    if zones:
        return zones[0]
    if not ignore_missing:
        raise exceptions.DNSZoneNotFound(zone_name=zone_name)