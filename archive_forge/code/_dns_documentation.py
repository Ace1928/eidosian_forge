from openstack.cloud import _utils
from openstack.dns.v2._proxy import Proxy
from openstack import exceptions
from openstack import resource
Delete a recordset.

        :param zone: Name, ID or :class:`openstack.dns.v2.zone.Zone` instance
            of the zone managing the recordset.
        :param name_or_id: Name or ID of the recordset being deleted.

        :returns: True if delete succeeded, False otherwise.
        :raises: :class:`~openstack.exceptions.SDKException` on operation
            error.
        