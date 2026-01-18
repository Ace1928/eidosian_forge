import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
class PropListDict(dict):
    """ a Dictionary which fetches values (InfoSvnCommand instances) lazily"""

    def __init__(self, path, keynames):
        dict.__init__(self, [(x, None) for x in keynames])
        self.path = path

    def __getitem__(self, key):
        value = dict.__getitem__(self, key)
        if value is None:
            value = self.path.propget(key)
            dict.__setitem__(self, key, value)
        return value