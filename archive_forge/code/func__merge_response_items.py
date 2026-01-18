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
def _merge_response_items(self, list_name, response_list):
    """
        Take a list of API responses ("item"-portion only) and combine them.

        Helper function to combine multiple aggegrated responses into a single
        dictionary that resembles an API response.

        Note: keys that don't have a 'list_name" key (including warnings)
        are omitted.

        :param   list_name: Name of list in dict.  Practically, this is
                          the name of the API called (e.g. 'disks').
        :type    list_name: ``str``

        :param   response_list: list of API responses (e.g. resp['items']).
                                Each entry in the list is the result of a
                                single API call.  Expected format is:
                                [ { items: {
                                             key1: { api_name:[]},
                                             key2: { api_name:[]}
                                           }}, ... ]
        :type    response_list: ``dict``

        :return: dict in the format of:
                 { items: {key: {api_name:[]}, key2: {api_name:[]}} }
                 ex: { items: {
                         'us-east1-a': {'disks': []},
                         'us-east1-b': {'disks': []}
                         }}
        :rtype:  ``dict``
        """
    merged_items = {}
    for resp in response_list:
        for k, v in resp.get('items', {}).items():
            if list_name in v:
                merged_items.setdefault(k, {}).setdefault(list_name, [])
                merged_items[k][list_name] += v[list_name]
    return {'items': merged_items}