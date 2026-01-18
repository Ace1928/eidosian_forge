import json
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType
from libcloud.utils.py3 import httplib
from libcloud.common.digitalocean import DigitalOcean_v2_BaseDriver, DigitalOcean_v2_Connection

        Delete a record.

        :param record: Record to delete.
        :type  record: :class:`Record`

        :rtype: ``bool``
        