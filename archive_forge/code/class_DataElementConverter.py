import re
from abc import ABCMeta
from copy import copy
from itertools import count
from typing import TYPE_CHECKING, cast, overload, Any, Dict, List, Iterator, \
from elementpath import XPathContext, XPath2Parser, build_node_tree, protocols
from elementpath.etree import etree_tostring
from .exceptions import XMLSchemaAttributeError, XMLSchemaTypeError, XMLSchemaValueError
from .aliases import ElementType, XMLSourceType, NamespacesType, BaseXsdType, DecodeType
from .helpers import get_namespace, get_prefixed_qname, local_name, raw_xml_encode
from .converters import ElementData, XMLSchemaConverter
from .resources import XMLResource
from . import validators
class DataElementConverter(XMLSchemaConverter):
    """
    XML Schema based converter class for DataElement objects.

    :param namespaces: a dictionary map from namespace prefixes to URI.
    :param data_element_class: MutableSequence subclass to use for decoded data.     Default is `DataElement`.
    :param map_attribute_names: define if map the names of attributes to prefixed     form. Defaults to `True`. If `False` the names are kept to extended format.
    """
    __slots__ = ('data_element_class', 'map_attribute_names')

    def __init__(self, namespaces: Optional[NamespacesType]=None, data_element_class: Optional[Type['DataElement']]=None, map_attribute_names: bool=True, **kwargs: Any) -> None:
        if data_element_class is None:
            self.data_element_class = DataElement
        else:
            self.data_element_class = data_element_class
        self.map_attribute_names = map_attribute_names
        kwargs.update(attr_prefix='', text_key='', cdata_prefix='')
        super(DataElementConverter, self).__init__(namespaces, **kwargs)

    @property
    def lossy(self) -> bool:
        return False

    @property
    def losslessly(self) -> bool:
        return True

    def copy(self, **kwargs: Any) -> 'DataElementConverter':
        obj = cast(DataElementConverter, super().copy(**kwargs))
        obj.data_element_class = kwargs.get('data_element_class', self.data_element_class)
        return obj

    def get_data_element(self, data: ElementData, xsd_element: 'XsdElement', xsd_type: Optional[BaseXsdType]=None) -> DataElement:
        return self.data_element_class(tag=data.tag, value=data.text, nsmap=self.namespaces, xsd_element=xsd_element, xsd_type=xsd_type)

    def element_decode(self, data: ElementData, xsd_element: 'XsdElement', xsd_type: Optional[BaseXsdType]=None, level: int=0) -> 'DataElement':
        data_element = self.get_data_element(data, xsd_element, xsd_type)
        if self.map_attribute_names:
            data_element.attrib.update(self.map_attributes(data.attributes))
        elif data.attributes:
            data_element.attrib.update(data.attributes)
        if (xsd_type or xsd_element.type).model_group is not None:
            for name, value, _ in self.map_content(data.content):
                if not name.isdigit():
                    data_element.append(value)
                else:
                    try:
                        data_element[-1].tail = value
                    except IndexError:
                        data_element.value = value
        return data_element

    def element_encode(self, data_element: 'DataElement', xsd_element: 'XsdElement', level: int=0) -> ElementData:
        self.namespaces.update(data_element.nsmap)
        if not xsd_element.is_matching(data_element.tag):
            raise XMLSchemaValueError('Unmatched tag')
        attributes = {self.unmap_qname(k, xsd_element.attributes): v for k, v in data_element.attrib.items()}
        data_len = len(data_element)
        if not data_len:
            return ElementData(data_element.tag, data_element.value, None, attributes)
        content: List[Tuple[Union[str, int], Any]] = []
        cdata_num = count(1)
        if data_element.value is not None:
            content.append((next(cdata_num), data_element.value))
        for e in data_element:
            content.append((e.tag, e))
            if e.tail is not None:
                content.append((next(cdata_num), e.tail))
        return ElementData(data_element.tag, None, content, attributes)