import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re

    Acts like SequenceMatcher, but tries not to find very small equal
    blocks amidst large spans of changes
    