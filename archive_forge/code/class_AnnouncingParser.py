import cProfile
from io import BytesIO
from html.parser import HTMLParser
import bs4
from bs4 import BeautifulSoup, __version__
from bs4.builder import builder_registry
import os
import pstats
import random
import tempfile
import time
import traceback
import sys
import cProfile
class AnnouncingParser(HTMLParser):
    """Subclass of HTMLParser that announces parse events, without doing
    anything else.

    You can use this to get a picture of how html.parser sees a given
    document. The easiest way to do this is to call `htmlparser_trace`.
    """

    def _p(self, s):
        print(s)

    def handle_starttag(self, name, attrs):
        self._p('%s START' % name)

    def handle_endtag(self, name):
        self._p('%s END' % name)

    def handle_data(self, data):
        self._p('%s DATA' % data)

    def handle_charref(self, name):
        self._p('%s CHARREF' % name)

    def handle_entityref(self, name):
        self._p('%s ENTITYREF' % name)

    def handle_comment(self, data):
        self._p('%s COMMENT' % data)

    def handle_decl(self, data):
        self._p('%s DECL' % data)

    def unknown_decl(self, data):
        self._p('%s UNKNOWN-DECL' % data)

    def handle_pi(self, data):
        self._p('%s PI' % data)