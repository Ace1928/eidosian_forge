from openstack.dns.v2 import floating_ip as _fip
from openstack.dns.v2 import recordset as _rs
from openstack.dns.v2 import zone as _zone
from openstack.dns.v2 import zone_export as _zone_export
from openstack.dns.v2 import zone_import as _zone_import
from openstack.dns.v2 import zone_share as _zone_share
from openstack.dns.v2 import zone_transfer as _zone_transfer
from openstack import proxy
def abandon_zone(self, zone, **attrs):
    """Abandon Zone

        :param zone: The value can be the ID of a zone to be abandoned
            or a :class:`~openstack.dns.v2.zone_export.ZoneExport` instance.

        :returns: None
        """
    zone = self._get_resource(_zone.Zone, zone)
    return zone.abandon(self)