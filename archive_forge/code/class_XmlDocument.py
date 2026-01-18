import json
import copy
from io import IOBase, TextIOBase
from typing import Any, Dict, List, Optional, Type, Union, Tuple, \
from elementpath.etree import ElementTree, etree_tostring
from .exceptions import XMLSchemaTypeError, XMLSchemaValueError, XMLResourceError
from .names import XSD_NAMESPACE, XSI_TYPE, XSD_SCHEMA
from .aliases import ElementType, XMLSourceType, NamespacesType, LocationsType, \
from .helpers import get_extended_qname, is_etree_document
from .resources import fetch_schema_locations, XMLResource
from .validators import XMLSchema10, XMLSchemaBase, XMLSchemaValidationError
class XmlDocument(XMLResource):
    """
    An XML document bound with its schema. If no schema is get from the provided
    context and validation argument is 'skip' the XML document is associated with
    a generic schema, otherwise a ValueError is raised.

    :param source: a string containing XML data or a file path or a URL or a     file like object or an ElementTree or an Element.
    :param schema: can be a :class:`xmlschema.XMLSchema` instance or a file-like     object or a file path or a URL of a resource or a string containing the XSD schema.
    :param cls: class to use for building the schema instance (for default     :class:`XMLSchema10` is used).
    :param validation: the XSD validation mode to use for validating the XML document,     that can be 'strict' (default), 'lax' or 'skip'.
    :param namespaces: is an optional mapping from namespace prefix to URI.
    :param locations: resource location hints, that can be a dictionary or a     sequence of couples (namespace URI, resource URL).
    :param base_url: the base URL for base :class:`xmlschema.XMLResource` initialization.
    :param allow: the security mode for base :class:`xmlschema.XMLResource` initialization.
    :param defuse: the defuse mode for base :class:`xmlschema.XMLResource` initialization.
    :param timeout: the timeout for base :class:`xmlschema.XMLResource` initialization.
    :param lazy: the lazy mode for base :class:`xmlschema.XMLResource` initialization.
    :param use_location_hints: for default, in case a schema instance has     to be built, uses also schema locations hints provided within XML data.     Set this option to `False` to ignore these schema location hints.
    """
    schema: Optional[XMLSchemaBase] = None
    _fallback_schema: Optional[XMLSchemaBase] = None
    validation: str = 'skip'
    namespaces: Optional[NamespacesType] = None
    errors: Union[Tuple[()], List[XMLSchemaValidationError]] = ()

    def __init__(self, source: XMLSourceType, schema: Optional[Union[XMLSchemaBase, SchemaSourceType]]=None, cls: Optional[Type[XMLSchemaBase]]=None, validation: str='strict', namespaces: Optional[NamespacesType]=None, locations: Optional[LocationsType]=None, base_url: Optional[str]=None, allow: str='all', defuse: str='remote', timeout: int=300, lazy: LazyType=False, use_location_hints: bool=True) -> None:
        if cls is None:
            cls = XMLSchema10
        self.validation = validation
        self._namespaces = namespaces
        super(XmlDocument, self).__init__(source, base_url, allow, defuse, timeout, lazy)
        if isinstance(schema, XMLSchemaBase) and self.namespace in schema.maps.namespaces:
            self.schema = schema
        elif schema is not None and (not isinstance(schema, XMLSchemaBase)):
            self.schema = cls(source=schema, locations=locations, base_url=base_url, allow=allow, defuse=defuse, timeout=timeout)
        else:
            if use_location_hints:
                try:
                    schema_location, locations = fetch_schema_locations(self, locations=locations, base_url=base_url)
                except ValueError:
                    pass
                else:
                    self.schema = cls(source=schema_location, locations=locations, allow=allow, defuse=defuse, timeout=timeout)
            if self.schema is None:
                if XSI_TYPE in self._root.attrib:
                    self.schema = get_dummy_schema(self._root.tag, cls)
                elif validation != 'skip':
                    msg = 'cannot get a schema for XML data, provide a schema argument'
                    raise XMLSchemaValueError(msg)
                else:
                    self._fallback_schema = get_dummy_schema(self._root.tag, cls)
        if self.schema is None:
            pass
        elif validation == 'strict':
            self.schema.validate(self, namespaces=self.namespaces)
        elif validation == 'lax':
            self.errors = [e for e in self.schema.iter_errors(self, namespaces=self.namespaces)]
        elif validation != 'skip':
            raise XMLSchemaValueError('%r is not a validation mode' % validation)

    def parse(self, source: XMLSourceType, lazy: LazyType=False) -> None:
        super(XmlDocument, self).parse(source, lazy)
        self.namespaces = self.get_namespaces()
        if self.schema is None:
            pass
        elif self.validation == 'strict':
            self.schema.validate(self, namespaces=self.namespaces)
        elif self.validation == 'lax':
            self.errors = [e for e in self.schema.iter_errors(self, namespaces=self.namespaces)]

    def get_namespaces(self, namespaces: Optional[NamespacesType]=None, root_only: Optional[bool]=None) -> NamespacesType:
        if not self._namespaces:
            _namespaces = namespaces
        elif not namespaces:
            _namespaces = self._namespaces
        else:
            _namespaces = copy.copy(self._namespaces)
            _namespaces.update(namespaces)
        return super().get_namespaces(_namespaces, root_only)

    def getroot(self) -> ElementType:
        """Get the root element of the XML document."""
        return self._root

    def get_etree_document(self) -> Any:
        """
        The resource as ElementTree XML document. If the resource is lazy
        raises a resource error.
        """
        if is_etree_document(self._source):
            return self._source
        elif self._lazy:
            raise XMLResourceError('cannot create an ElementTree instance from a lazy XML resource')
        elif hasattr(self._root, 'nsmap'):
            return self._root.getroottree()
        else:
            return ElementTree.ElementTree(self._root)

    def decode(self, **kwargs: Any) -> DecodeType[Any]:
        """
        Decode the XML document to a nested Python dictionary.

        :param kwargs: options for the decode/to_dict method of the schema instance.
        """
        if 'validation' not in kwargs:
            kwargs['validation'] = self.validation
        if 'namespaces' not in kwargs:
            kwargs['namespaces'] = self.namespaces
        schema = self.schema or self._fallback_schema
        if schema is None:
            return None
        obj = schema.to_dict(self, **kwargs)
        return obj[0] if isinstance(obj, tuple) else obj

    def to_json(self, fp: Optional[IO[str]]=None, json_options: Optional[Dict[str, Any]]=None, **kwargs: Any) -> JsonDecodeType:
        """
        Converts loaded XML data to a JSON string or file.

        :param fp: can be a :meth:`write()` supporting file-like object.
        :param json_options: a dictionary with options for the JSON deserializer.
        :param kwargs: options for the decode/to_dict method of the schema instance.
        """
        if json_options is None:
            json_options = {}
        path = kwargs.pop('path', None)
        if 'validation' not in kwargs:
            kwargs['validation'] = self.validation
        if 'namespaces' not in kwargs:
            kwargs['namespaces'] = self.namespaces
        if 'decimal_type' not in kwargs:
            kwargs['decimal_type'] = float
        errors: List[XMLSchemaValidationError] = []
        if path is None and self._lazy and ('cls' not in json_options):
            json_options['cls'] = get_lazy_json_encoder(errors)
            kwargs['lazy_decode'] = True
        schema = self.schema or self._fallback_schema
        if schema is not None:
            obj = schema.decode(self, path=path, **kwargs)
        else:
            obj = None
        if isinstance(obj, tuple):
            if fp is not None:
                json.dump(obj[0], fp, **json_options)
                obj[1].extend(errors)
                return tuple(obj[1])
            else:
                result = json.dumps(obj[0], **json_options)
                obj[1].extend(errors)
                return (result, tuple(obj[1]))
        elif fp is not None:
            json.dump(obj, fp, **json_options)
            return None if not errors else tuple(errors)
        else:
            result = json.dumps(obj, **json_options)
            return result if not errors else (result, tuple(errors))

    def write(self, file: Union[str, TextIO, BinaryIO], encoding: str='us-ascii', xml_declaration: bool=False, default_namespace: Optional[str]=None, method: str='xml') -> None:
        """Serialize an XML resource to a file. Cannot be used with lazy resources."""
        if self._lazy:
            raise XMLResourceError('cannot serialize a lazy XML resource')
        kwargs: Dict[str, Any] = {'xml_declaration': xml_declaration, 'encoding': encoding, 'method': method}
        if not default_namespace:
            kwargs['namespaces'] = self.namespaces
        else:
            namespaces: Optional[Dict[Optional[str], str]]
            if self.namespaces is None:
                namespaces = {}
            else:
                namespaces = {k: v for k, v in self.namespaces.items()}
            if hasattr(self._root, 'nsmap'):
                namespaces[None] = default_namespace
            else:
                namespaces[''] = default_namespace
            kwargs['namespaces'] = namespaces
        _string = etree_tostring(self._root, **kwargs)
        if isinstance(file, str):
            if isinstance(_string, str):
                with open(file, 'w', encoding='utf-8') as fp:
                    fp.write(_string)
            else:
                with open(file, 'wb') as _fp:
                    _fp.write(_string)
        elif isinstance(file, TextIOBase):
            if isinstance(_string, bytes):
                file.write(_string.decode('utf-8'))
            else:
                file.write(_string)
        elif isinstance(file, IOBase):
            if isinstance(_string, str):
                file.write(_string.encode('utf-8'))
            else:
                file.write(_string)
        else:
            msg = "unexpected type %r for 'file' argument"
            raise XMLSchemaTypeError(msg % type(file))