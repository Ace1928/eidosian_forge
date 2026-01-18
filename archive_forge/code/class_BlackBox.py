import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class BlackBox(Container):
    """A container that does not output anything"""

    def __init__(self):
        self.parser = LoneCommand()
        self.output = EmptyOutput()
        self.contents = []