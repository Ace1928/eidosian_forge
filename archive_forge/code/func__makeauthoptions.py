import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
def _makeauthoptions(self):
    if self.auth is None:
        return ''
    return self.auth.makecmdoptions()