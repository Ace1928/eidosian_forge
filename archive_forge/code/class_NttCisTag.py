import re
import xml.etree.ElementTree as etree
from io import BytesIO
from copy import deepcopy
from time import sleep
from base64 import b64encode
from typing import Dict
from functools import wraps
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class NttCisTag:
    """
    A representation of a Tag in NTTCIS
    A Tag first must have a Tag Key, then an asset is tag with
    a key and an option value.  Tags can be queried later to filter assets
    and also show up on usage report if so desired.
    """

    def __init__(self, asset_type, asset_id, asset_name, datacenter, key, value):
        """
        Initialize an instance of :class:`NttCisTag`

        :param asset_type: The type of asset.  Current asset types:
                           SERVER, VLAN, NETWORK_DOMAIN, CUSTOMER_IMAGE,
                           PUBLIC_IP_BLOCK, ACCOUNT
        :type  asset_type: ``str``

        :param asset_id: The GUID of the asset that is tagged
        :type  asset_id: ``str``

        :param asset_name: The name of the asset that is tagged
        :type  asset_name: ``str``

        :param datacenter: The short datacenter name of the tagged asset
        :type  datacenter: ``str``

        :param key: The tagged key
        :type  key: :class:`NttCisTagKey`

        :param value: The tagged value
        :type  value: ``None`` or ``str``
        """
        self.asset_type = asset_type
        self.asset_id = asset_id
        self.asset_name = asset_name
        self.datacenter = datacenter
        self.key = key
        self.value = value

    def __repr__(self):
        return '<NttCisTag: asset_name=%s, tag_name=%s, value=%s>' % (self.asset_name, self.key.name, self.value)