import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class LayoutConfig(object):
    """Configuration class from elyxer.config file"""
    groupable = {'allowed': ['StringContainer', 'Constant', 'TaggedText', 'Align', 'TextFamily', 'EmphaticText', 'VersalitasText', 'BarredText', 'SizeText', 'ColorText', 'LangLine', 'Formula']}