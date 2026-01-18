import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
class NoDeletes(Exception):
    """ Raised when the document no longer contains any pending deletes
    (DEL_START/DEL_END) """