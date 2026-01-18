import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def _merge_element_contents(el):
    """
    Removes an element, but merges its contents into its place, e.g.,
    given <p>Hi <i>there!</i></p>, if you remove the <i> element you get
    <p>Hi there!</p>
    """
    parent = el.getparent()
    text = el.text or ''
    if el.tail:
        if not len(el):
            text += el.tail
        elif el[-1].tail:
            el[-1].tail += el.tail
        else:
            el[-1].tail = el.tail
    index = parent.index(el)
    if text:
        if index == 0:
            previous = None
        else:
            previous = parent[index - 1]
        if previous is None:
            if parent.text:
                parent.text += text
            else:
                parent.text = text
        elif previous.tail:
            previous.tail += text
        else:
            previous.tail = text
    parent[index:index + 1] = el.getchildren()