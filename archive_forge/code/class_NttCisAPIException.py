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
class NttCisAPIException(LibcloudError):

    def __init__(self, code, msg, driver):
        self.code = code
        self.msg = msg
        self.driver = driver

    def __str__(self):
        return '{}: {}'.format(self.code, self.msg)

    def __repr__(self):
        return "<NttCisAPIException: code='{}', msg='{}'>".format(self.code, self.msg)