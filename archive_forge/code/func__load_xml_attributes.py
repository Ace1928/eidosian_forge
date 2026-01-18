import itertools
import os
import re
import sys
import textwrap
import types
from collections import OrderedDict, defaultdict
from itertools import zip_longest
from operator import itemgetter
from pprint import pprint
from nltk.corpus.reader import XMLCorpusReader, XMLCorpusView
from nltk.util import LazyConcatenation, LazyIteratorList, LazyMap
def _load_xml_attributes(self, d, elt):
    """
        Extracts a subset of the attributes from the given element and
        returns them in a dictionary.

        :param d: A dictionary in which to store the attributes.
        :type d: dict
        :param elt: An ElementTree Element
        :type elt: Element
        :return: Returns the input dict ``d`` possibly including attributes from ``elt``
        :rtype: dict
        """
    d = type(d)(d)
    try:
        attr_dict = elt.attrib
    except AttributeError:
        return d
    if attr_dict is None:
        return d
    ignore_attrs = ['xsi', 'schemaLocation', 'xmlns', 'bgColor', 'fgColor']
    for attr in attr_dict:
        if any((attr.endswith(x) for x in ignore_attrs)):
            continue
        val = attr_dict[attr]
        if val.isdigit():
            d[attr] = int(val)
        else:
            d[attr] = val
    return d