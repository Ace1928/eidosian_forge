import json, sys
from xml.dom import minidom
import plistlib
import logging
class PListResponseParser(ResponseParser):

    def __init__(self):
        self.meta = 'text/xml'

    def parse(self, xml, pvars):
        if sys.version_info >= (3, 0):
            pl = plistlib.readPlistFromBytes(xml.encode())
        else:
            pl = plistlib.readPlistFromString(xml)
        parsed = {}
        pvars = self.getVars(pvars)
        for k, v in pvars.items():
            parsed[k] = pl[k] if k in pl else None
        return parsed