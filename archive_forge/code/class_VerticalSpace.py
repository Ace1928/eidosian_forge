import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class VerticalSpace(Container):
    """An inset that contains a vertical space."""

    def __init__(self):
        self.parser = InsetParser()
        self.output = FixedOutput()

    def process(self):
        """Set the correct tag"""
        self.type = self.header[2]
        if self.type not in StyleConfig.vspaces:
            self.output = TaggedOutput().settag('div class="vspace" style="height: ' + self.type + ';"', True)
            return
        self.html = [StyleConfig.vspaces[self.type]]