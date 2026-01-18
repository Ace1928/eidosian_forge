from lxml.etree import XPath, ElementBase
from lxml.html import fromstring, XHTML_NAMESPACE
from lxml.html import _forms_xpath, _options_xpath, _nons, _transform_result
from lxml.html import defs
import copy
def _find_elements_for_name(form, name, error):
    if name is None:
        yield (form, error)
        return
    if name.startswith('#'):
        el = form.get_element_by_id(name[1:])
        if el is not None:
            yield (el, error)
        return
    els = _name_xpath(form, name=name)
    if not els:
        return
    if not isinstance(error, (list, tuple)):
        yield (els[0], error)
        return
    for el, err in zip(els, error):
        if err is None:
            continue
        yield (el, err)