from collections.abc import MutableMapping, MutableSequence
from typing import TYPE_CHECKING, Any, Optional, List, Dict, Type, Tuple
from ..exceptions import XMLSchemaTypeError, XMLSchemaValueError
from ..aliases import NamespacesType, BaseXsdType
from .default import ElementData, XMLSchemaConverter

    XML Schema based converter class for columnar formats.

    :param namespaces: map from namespace prefixes to URI.
    :param dict_class: dictionary class to use for decoded data. Default is `dict`.
    :param list_class: list class to use for decoded data. Default is `list`.
    :param attr_prefix: used as separator string for renaming the decoded attributes.     Can be the empty string (the default) or a single/double underscore.
    