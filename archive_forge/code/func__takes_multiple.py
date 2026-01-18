from lxml.etree import XPath, ElementBase
from lxml.html import fromstring, XHTML_NAMESPACE
from lxml.html import _forms_xpath, _options_xpath, _nons, _transform_result
from lxml.html import defs
import copy
def _takes_multiple(input):
    if _nons(input.tag) == 'select' and input.get('multiple'):
        return True
    type = input.get('type', '').lower()
    if type in ('radio', 'checkbox'):
        return True
    return False