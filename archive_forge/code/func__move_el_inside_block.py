import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def _move_el_inside_block(el, tag):
    """ helper for _fixup_ins_del_tags; actually takes the <ins> etc tags
    and moves them inside any block-level tags.  """
    for child in el:
        if _contains_block_level_tag(child):
            break
    else:
        children_tag = etree.Element(tag)
        children_tag.text = el.text
        el.text = None
        children_tag.extend(list(el))
        el[:] = [children_tag]
        return
    for child in list(el):
        if _contains_block_level_tag(child):
            _move_el_inside_block(child, tag)
            if child.tail:
                tail_tag = etree.Element(tag)
                tail_tag.text = child.tail
                child.tail = None
                el.insert(el.index(child) + 1, tail_tag)
        else:
            child_tag = etree.Element(tag)
            el.replace(child, child_tag)
            child_tag.append(child)
    if el.text:
        text_tag = etree.Element(tag)
        text_tag.text = el.text
        el.text = None
        el.insert(0, text_tag)