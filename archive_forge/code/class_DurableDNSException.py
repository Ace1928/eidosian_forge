import re
from typing import Dict, List
from xml.etree import ElementTree as ET  # noqa
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
class DurableDNSException(Exception):

    def __init__(self, code, message):
        self.code = code
        self.message = message
        self.args = (code, message)

    def __str__(self):
        return '{} {}'.format(self.code, self.message)

    def __repr__(self):
        return 'DurableDNSException {} {}'.format(self.code, self.message)