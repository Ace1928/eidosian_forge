from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.common.gandi import GandiResponse, BaseGandiDriver, GandiConnection
class NewZoneVersion:
    """
    Changes to a zone in the Gandi DNS service need to be wrapped in a new
    version object. The changes are made to the new version, then that
    version is made active.

    In effect, this is a transaction.

    Any calls made inside this context manager will be applied to a new version
    id. If your changes are successful (and only if they are successful) they
    are activated.
    """

    def __init__(self, driver, zone):
        self.driver = driver
        self.connection = driver.connection
        self.zone = zone

    def __enter__(self):
        zid = int(self.zone.id)
        self.connection.set_context({'zone_id': self.zone.id})
        vid = self.connection.request('domain.zone.version.new', zid).object
        self.vid = vid
        return vid

    def __exit__(self, type, value, traceback):
        if not traceback:
            zid = int(self.zone.id)
            con = self.connection
            con.set_context({'zone_id': self.zone.id})
            con.request('domain.zone.version.set', zid, self.vid).object