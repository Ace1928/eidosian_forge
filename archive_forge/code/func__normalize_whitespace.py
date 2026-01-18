import re
import lxml
import lxml.etree
from lxml.html.clean import Cleaner
def _normalize_whitespace(text):
    return _whitespace.sub(' ', text.strip())