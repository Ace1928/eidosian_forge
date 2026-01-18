import json, sys
from xml.dom import minidom
import plistlib
import logging
class JSONResponseParser(ResponseParser):

    def __init__(self):
        self.meta = 'text/json'

    def parse(self, jsonData, pvars):
        d = json.loads(jsonData)
        pvars = self.getVars(pvars)
        parsed = {}
        for k, v in pvars.items():
            parsed[k] = self.query(d, v)
        return parsed

    def query(self, d, key):
        keys = key.split('.', 1)
        currKey = keys[0]
        if currKey in d:
            item = d[currKey]
            if len(keys) == 1:
                return item
            if type(item) is dict:
                return self.query(item, keys[1])
            elif type(item) is list:
                output = []
                for i in item:
                    output.append(self.query(i, keys[1]))
                return output
            else:
                return None