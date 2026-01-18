from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.types import ProviderError, MalformedResponseError
from libcloud.common.pointdns import PointDNSConnection
from libcloud.common.exceptions import BaseHTTPError
class MailRedirect:
    """
    Point DNS mail redirect.
    """

    def __init__(self, id, source, destination, zone, driver):
        """
        :param id: MailRedirect id.
        :type id: ``str``

        :param source: The source address of mail redirect.
        :type source: ``str``

        :param destination: The destination address of mail redirect.
        :type destination: ``str``

        :param zone: Zone where mail redirect belongs.
        :type  zone: :class:`Zone`

        :param driver: DNSDriver instance.
        :type driver: :class:`DNSDriver`
        """
        self.id = str(id) if id else None
        self.source = source
        self.destination = destination
        self.zone = zone
        self.driver = driver

    def update(self, destination, source=None):
        return self.driver.ex_update_mail_redirect(mail_r=self, destination=destination, source=None)

    def delete(self):
        return self.driver.ex_delete_mail_redirect(mail_r=self)

    def __repr__(self):
        return '<PointDNSMailRedirect: source={}, destination={},zone={} ...>'.format(self.source, self.destination, self.zone.id)