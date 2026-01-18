import lxml.etree as ET
from functools import partial
def add_cdata(elem, cdata):
    if elem.text:
        raise ValueError("Can't add a CDATA section. Element already has some text: %r" % elem.text)
    elem.text = cdata