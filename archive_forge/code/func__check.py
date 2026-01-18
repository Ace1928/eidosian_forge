from lxml.etree import XPath, ElementBase
from lxml.html import fromstring, XHTML_NAMESPACE
from lxml.html import _forms_xpath, _options_xpath, _nons, _transform_result
from lxml.html import defs
import copy
def _check(el, check):
    if check:
        el.set('checked', '')
    elif 'checked' in el.attrib:
        del el.attrib['checked']