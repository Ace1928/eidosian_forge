from __future__ import annotations
import io
from os import PathLike
from typing import (
import warnings
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import is_list_like
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import (
from pandas.io.parsers import TextParser
class _EtreeFrameParser(_XMLFrameParser):
    """
    Internal class to parse XML into DataFrames with the Python
    standard library XML module: `xml.etree.ElementTree`.
    """

    def parse_data(self) -> list[dict[str, str | None]]:
        from xml.etree.ElementTree import iterparse
        if self.stylesheet is not None:
            raise ValueError('To use stylesheet, you need lxml installed and selected as parser.')
        if self.iterparse is None:
            self.xml_doc = self._parse_doc(self.path_or_buffer)
            elems = self._validate_path()
        self._validate_names()
        xml_dicts: list[dict[str, str | None]] = self._parse_nodes(elems) if self.iterparse is None else self._iterparse_nodes(iterparse)
        return xml_dicts

    def _validate_path(self) -> list[Any]:
        """
        Notes
        -----
        ``etree`` supports limited ``XPath``. If user attempts a more complex
        expression syntax error will raise.
        """
        msg = 'xpath does not return any nodes or attributes. Be sure to specify in `xpath` the parent nodes of children and attributes to parse. If document uses namespaces denoted with xmlns, be sure to define namespaces and use them in xpath.'
        try:
            elems = self.xml_doc.findall(self.xpath, namespaces=self.namespaces)
            children = [ch for el in elems for ch in el.findall('*')]
            attrs = {k: v for el in elems for k, v in el.attrib.items()}
            if elems is None:
                raise ValueError(msg)
            if elems is not None:
                if self.elems_only and children == []:
                    raise ValueError(msg)
                if self.attrs_only and attrs == {}:
                    raise ValueError(msg)
                if children == [] and attrs == {}:
                    raise ValueError(msg)
        except (KeyError, SyntaxError):
            raise SyntaxError('You have used an incorrect or unsupported XPath expression for etree library or you used an undeclared namespace prefix.')
        return elems

    def _validate_names(self) -> None:
        children: list[Any]
        if self.names:
            if self.iterparse:
                children = self.iterparse[next(iter(self.iterparse))]
            else:
                parent = self.xml_doc.find(self.xpath, namespaces=self.namespaces)
                children = parent.findall('*') if parent is not None else []
            if is_list_like(self.names):
                if len(self.names) < len(children):
                    raise ValueError('names does not match length of child elements in xpath.')
            else:
                raise TypeError(f'{type(self.names).__name__} is not a valid type for names')

    def _parse_doc(self, raw_doc: FilePath | ReadBuffer[bytes] | ReadBuffer[str]) -> Element:
        from xml.etree.ElementTree import XMLParser, parse
        handle_data = get_data_from_filepath(filepath_or_buffer=raw_doc, encoding=self.encoding, compression=self.compression, storage_options=self.storage_options)
        with preprocess_data(handle_data) as xml_data:
            curr_parser = XMLParser(encoding=self.encoding)
            document = parse(xml_data, parser=curr_parser)
        return document.getroot()