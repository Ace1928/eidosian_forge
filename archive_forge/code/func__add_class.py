from lxml.etree import XPath, ElementBase
from lxml.html import fromstring, XHTML_NAMESPACE
from lxml.html import _forms_xpath, _options_xpath, _nons, _transform_result
from lxml.html import defs
import copy
def _add_class(el, class_name):
    if el.get('class'):
        el.set('class', el.get('class') + ' ' + class_name)
    else:
        el.set('class', class_name)