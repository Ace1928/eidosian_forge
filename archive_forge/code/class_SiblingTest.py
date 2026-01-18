from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
class SiblingTest(SoupTest):

    def setup_method(self):
        markup = '<html>\n                    <span id="1">\n                     <span id="1.1"></span>\n                    </span>\n                    <span id="2">\n                     <span id="2.1"></span>\n                    </span>\n                    <span id="3">\n                     <span id="3.1"></span>\n                    </span>\n                    <span id="4"></span>\n                    </html>'
        markup = re.compile('\\n\\s*').sub('', markup)
        self.tree = self.soup(markup)