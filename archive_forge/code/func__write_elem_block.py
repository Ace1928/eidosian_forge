import re
import warnings
from itertools import chain
from xml.etree import ElementTree
from xml.sax.saxutils import XMLGenerator, escape
from Bio import BiopythonParserWarning
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _write_elem_block(self, block_name, map_name, obj, opt_dict=None):
    """Write sibling XML elements (PRIVATE).

        :param block_name: common element name prefix
        :type block_name: string
        :param map_name: name of mapping between element and attribute names
        :type map_name: string
        :param obj: object whose attribute value will be used
        :type obj: object
        :param opt_dict: custom element-attribute mapping
        :type opt_dict: dictionary {string: string}

        """
    if opt_dict is None:
        opt_dict = {}
    for elem, attr in _WRITE_MAPS[map_name]:
        elem = block_name + elem
        try:
            content = str(getattr(obj, attr))
        except AttributeError:
            if elem not in _DTD_OPT:
                raise ValueError(f'Element {elem!r} (attribute {attr!r}) not found')
        else:
            if elem in opt_dict:
                content = opt_dict[elem]
            self.xml.simpleElement(elem, content)