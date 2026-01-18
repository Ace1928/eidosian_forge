import time
from libcloud.compute.base import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.gridscale import GridscaleBaseDriver, GridscaleConnection
from libcloud.compute.providers import Provider
def _get_response_dict(self, raw_response):
    """

        Get the actual response dictionary.

        :param raw_response: Nested dictionary.
        :type raw_response: ``dict``

        :return: Not-nested dictionary.
        :rtype: ``dict``
        """
    return list(raw_response.object.values())[0]