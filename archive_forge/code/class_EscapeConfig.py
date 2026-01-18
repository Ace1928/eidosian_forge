import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class EscapeConfig(object):
    """Configuration class from elyxer.config file"""
    chars = {'\n': '', ' -- ': ' — ', ' --- ': ' — ', "'": '’', '`': '‘'}
    commands = {'\\InsetSpace \\space{}': '\xa0', '\\InsetSpace \\thinspace{}': '\u2009', '\\InsetSpace ~': '\xa0', '\\SpecialChar \\-': '', '\\SpecialChar \\@.': '.', '\\SpecialChar \\ldots{}': '…', '\\SpecialChar \\menuseparator': '\xa0▷\xa0', '\\SpecialChar \\nobreakdash-': '-', '\\SpecialChar \\slash{}': '/', '\\SpecialChar \\textcompwordmark{}': '', '\\backslash': '\\'}
    entities = {'&': '&amp;', '<': '&lt;', '>': '&gt;'}
    html = {'/>': '>'}
    iso885915 = {'\xa0': '&nbsp;', '\u2003': '&emsp;', '\u205f': '&#8197;'}
    nonunicode = {'\u205f': '\u2005'}