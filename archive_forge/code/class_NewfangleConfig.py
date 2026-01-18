import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class NewfangleConfig(object):
    """Configuration class from elyxer.config file"""
    constants = {'chunkref': 'chunkref{', 'endcommand': '}', 'endmark': '&gt;', 'startcommand': '\\', 'startmark': '=&lt;'}