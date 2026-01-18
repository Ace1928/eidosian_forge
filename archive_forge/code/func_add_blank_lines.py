import codecs
import re
from io import StringIO
from xml.etree.ElementTree import Element, ElementTree, SubElement, TreeBuilder
from nltk.data import PathPointer, find
def add_blank_lines(tree, blanks_before, blanks_between):
    """
    Add blank lines before all elements and subelements specified in blank_before.

    :param elem: toolbox data in an elementtree structure
    :type elem: ElementTree._ElementInterface
    :param blank_before: elements and subelements to add blank lines before
    :type blank_before: dict(tuple)
    """
    try:
        before = blanks_before[tree.tag]
        between = blanks_between[tree.tag]
    except KeyError:
        for elem in tree:
            if len(elem):
                add_blank_lines(elem, blanks_before, blanks_between)
    else:
        last_elem = None
        for elem in tree:
            tag = elem.tag
            if last_elem is not None and last_elem.tag != tag:
                if tag in before and last_elem is not None:
                    e = last_elem.getiterator()[-1]
                    e.text = (e.text or '') + '\n'
            elif tag in between:
                e = last_elem.getiterator()[-1]
                e.text = (e.text or '') + '\n'
            if len(elem):
                add_blank_lines(elem, blanks_before, blanks_between)
            last_elem = elem