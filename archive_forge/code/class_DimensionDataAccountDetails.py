from time import sleep
from base64 import b64encode
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class DimensionDataAccountDetails:
    """
    Dimension Data account class details
    """

    def __init__(self, user_name, full_name, first_name, last_name, email):
        self.user_name = user_name
        self.full_name = full_name
        self.first_name = first_name
        self.last_name = last_name
        self.email = email