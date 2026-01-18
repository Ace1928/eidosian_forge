import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
def _geturl(self):
    if getattr(self, '_url', None) is None:
        info = self.info()
        self._url = info.url
    assert isinstance(self._url, py.builtin._basestring)
    return self._url