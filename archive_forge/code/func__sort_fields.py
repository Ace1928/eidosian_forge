import codecs
import re
from io import StringIO
from xml.etree.ElementTree import Element, ElementTree, SubElement, TreeBuilder
from nltk.data import PathPointer, find
def _sort_fields(elem, orders_dicts):
    """sort the children of elem"""
    try:
        order = orders_dicts[elem.tag]
    except KeyError:
        pass
    else:
        tmp = sorted((((order.get(child.tag, 1000000000.0), i), child) for i, child in enumerate(elem)))
        elem[:] = [child for key, child in tmp]
    for child in elem:
        if len(child):
            _sort_fields(child, orders_dicts)