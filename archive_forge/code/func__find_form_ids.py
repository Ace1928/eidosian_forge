from lxml.etree import XPath, ElementBase
from lxml.html import fromstring, XHTML_NAMESPACE
from lxml.html import _forms_xpath, _options_xpath, _nons, _transform_result
from lxml.html import defs
import copy
def _find_form_ids(el):
    forms = _forms_xpath(el)
    if not forms:
        yield '(no forms)'
        return
    for index, form in enumerate(forms):
        if form.get('id'):
            if form.get('name'):
                yield ('%s or %s' % (form.get('id'), form.get('name')))
            else:
                yield form.get('id')
        elif form.get('name'):
            yield form.get('name')
        else:
            yield ('(unnamed form %s)' % index)