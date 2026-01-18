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
class EC2NodeLocation(NodeLocation):

    def __init__(self, id, name, country, driver, availability_zone):
        super().__init__(id, name, country, driver)
        self.availability_zone = availability_zone

    def __repr__(self):
        return '<EC2NodeLocation: id=%s, name=%s, country=%s, availability_zone=%s driver=%s>' % (self.id, self.name, self.country, self.availability_zone, self.driver.name)