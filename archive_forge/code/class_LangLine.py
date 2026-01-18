import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class LangLine(TaggedText):
    """A line with language information"""

    def process(self):
        """Only generate a span with lang info when the language is recognized."""
        lang = self.header[1]
        if not lang in TranslationConfig.languages:
            self.output = ContentsOutput()
            return
        isolang = TranslationConfig.languages[lang]
        self.output = TaggedOutput().settag('span lang="' + isolang + '"', False)