import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
def _escape_helper(text):
    text = str(text)
    if sys.platform != 'win32':
        text = str(text).replace('$', '\\$')
    return text