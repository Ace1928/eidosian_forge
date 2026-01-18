import time
from libcloud.compute.base import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.gridscale import GridscaleBaseDriver, GridscaleConnection
from libcloud.compute.providers import Provider
def _extract_values_to_dict(self, data, keys):
    """
        Extract extra values to dict.

        :param data: dict to extract values from.
        :type data: ``dict``
        :param keys: keys to extract
        :type keys: ``List``
        :return: dictionary containing extra values
        :rtype: ``dict``
        """
    result = {}
    for key in keys:
        if key == 'memory':
            result[key] = data[key] * 1024
        else:
            result[key] = data[key]
    return result