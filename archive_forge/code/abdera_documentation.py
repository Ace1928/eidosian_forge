from collections.abc import MutableMapping, MutableSequence
from typing import TYPE_CHECKING, Any, Optional, List, Dict, Type, Union
from ..exceptions import XMLSchemaValueError
from ..aliases import NamespacesType, BaseXsdType
from .default import ElementData, XMLSchemaConverter

    XML Schema based converter class for Abdera convention.

    ref: http://wiki.open311.org/JSON_and_XML_Conversion/#the-abdera-convention
    ref: https://cwiki.apache.org/confluence/display/ABDERA/JSON+Serialization

    :param namespaces: Map from namespace prefixes to URI.
    :param dict_class: Dictionary class to use for decoded data. Default is `dict`.
    :param list_class: List class to use for decoded data. Default is `list`.
    