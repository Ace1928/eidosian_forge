from lxml.etree import XPath, ElementBase
from lxml.html import fromstring, XHTML_NAMESPACE
from lxml.html import _forms_xpath, _options_xpath, _nons, _transform_result
from lxml.html import defs
import copy
def _find_form(el, form_id=None, form_index=None):
    if form_id is None and form_index is None:
        forms = _forms_xpath(el)
        for form in forms:
            return form
        raise FormNotFound('No forms in page')
    if form_id is not None:
        form = el.get_element_by_id(form_id)
        if form is not None:
            return form
        forms = _form_name_xpath(el, name=form_id)
        if forms:
            return forms[0]
        else:
            raise FormNotFound('No form with the name or id of %r (forms: %s)' % (id, ', '.join(_find_form_ids(el))))
    if form_index is not None:
        forms = _forms_xpath(el)
        try:
            return forms[form_index]
        except IndexError:
            raise FormNotFound('There is no form with the index %r (%i forms found)' % (form_index, len(forms)))