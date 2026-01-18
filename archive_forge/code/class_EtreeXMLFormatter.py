from __future__ import annotations
import codecs
import io
from typing import (
import warnings
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.missing import isna
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle
from pandas.io.xml import (
class EtreeXMLFormatter(_BaseXMLFormatter):
    """
    Class for formatting data in xml using Python standard library
    modules: `xml.etree.ElementTree` and `xml.dom.minidom`.
    """

    def _build_tree(self) -> bytes:
        from xml.etree.ElementTree import Element, SubElement, tostring
        self.root = Element(f'{self.prefix_uri}{self.root_name}', attrib=self._other_namespaces())
        for d in self.frame_dicts.values():
            elem_row = SubElement(self.root, f'{self.prefix_uri}{self.row_name}')
            if not self.attr_cols and (not self.elem_cols):
                self.elem_cols = list(d.keys())
                self._build_elems(d, elem_row)
            else:
                elem_row = self._build_attribs(d, elem_row)
                self._build_elems(d, elem_row)
        self.out_xml = tostring(self.root, method='xml', encoding=self.encoding, xml_declaration=self.xml_declaration)
        if self.pretty_print:
            self.out_xml = self._prettify_tree()
        if self.stylesheet is not None:
            raise ValueError('To use stylesheet, you need lxml installed and selected as parser.')
        return self.out_xml

    def _get_prefix_uri(self) -> str:
        from xml.etree.ElementTree import register_namespace
        uri = ''
        if self.namespaces:
            for p, n in self.namespaces.items():
                if isinstance(p, str) and isinstance(n, str):
                    register_namespace(p, n)
            if self.prefix:
                try:
                    uri = f'{{{self.namespaces[self.prefix]}}}'
                except KeyError:
                    raise KeyError(f'{self.prefix} is not included in namespaces')
            elif '' in self.namespaces:
                uri = f'{{{self.namespaces['']}}}'
            else:
                uri = ''
        return uri

    @cache_readonly
    def _sub_element_cls(self):
        from xml.etree.ElementTree import SubElement
        return SubElement

    def _prettify_tree(self) -> bytes:
        """
        Output tree for pretty print format.

        This method will pretty print xml with line breaks and indentation.
        """
        from xml.dom.minidom import parseString
        dom = parseString(self.out_xml)
        return dom.toprettyxml(indent='  ', encoding=self.encoding)