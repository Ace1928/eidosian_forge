import re
import copy
import time
import base64
import warnings
from typing import List
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, basestring, ensure_string
from libcloud.utils.xml import findall, findattr, findtext, fixxpath
from libcloud.common.aws import DEFAULT_SIGNATURE_VERSION, AWSBaseResponse, SignedAWSConnection
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date, parse_date_allow_empty
from libcloud.utils.publickey import get_pubkey_comment, get_pubkey_ssh2_fingerprint
from libcloud.compute.providers import Provider
from libcloud.compute.constants.ec2_region_details_partial import (
def _to_route_table(self, element, name=None):
    route_table_id = findtext(element=element, xpath='routeTableId', namespace=NAMESPACE)
    tags = self._get_resource_tags(element)
    extra = self._get_extra_dict(element, RESOURCE_EXTRA_ATTRIBUTES_MAP['route_table'])
    extra['tags'] = tags
    routes = self._to_routes(element, 'routeSet/item')
    subnet_associations = self._to_subnet_associations(element, 'associationSet/item')
    propagating_gateway_ids = []
    for el in element.findall(fixxpath(xpath='propagatingVgwSet/item', namespace=NAMESPACE)):
        propagating_gateway_ids.append(findtext(element=el, xpath='gatewayId', namespace=NAMESPACE))
    name = name if name else tags.get('Name', id)
    return EC2RouteTable(route_table_id, name, routes, subnet_associations, propagating_gateway_ids, extra=extra)