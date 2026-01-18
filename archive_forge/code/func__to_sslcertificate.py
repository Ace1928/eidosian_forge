import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def _to_sslcertificate(self, sslcertificate):
    """
        Return the SslCertificate object from the JSON-response.

        :param  sslcertificate:  Dictionary describing SslCertificate
        :type   sslcertificate: ``dict``

        :return:  Return SslCertificate object.
        :rtype: :class:`GCESslCertificate`
        """
    extra = {}
    if 'description' in sslcertificate:
        extra['description'] = sslcertificate['description']
    extra['selfLink'] = sslcertificate['selfLink']
    return GCESslCertificate(id=sslcertificate['id'], name=sslcertificate['name'], certificate=sslcertificate['certificate'], driver=self, extra=extra)