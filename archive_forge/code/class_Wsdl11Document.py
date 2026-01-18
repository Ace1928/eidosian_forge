import os
from ..exceptions import XMLSchemaException, XMLSchemaValueError
from ..names import XSD_NAMESPACE, WSDL_NAMESPACE, SOAP_NAMESPACE, \
from ..helpers import get_qname, local_name, get_extended_qname, get_prefixed_qname
from ..namespaces import NamespaceResourcesMap
from ..resources import fetch_resource
from ..documents import XmlDocument
from ..validators import XMLSchemaBase, XMLSchema10
class Wsdl11Document(XmlDocument):
    """
    Class for WSDL 1.1 documents.

    :param source: a string containing XML data or a file path or an URL or a     file like object or an ElementTree or an Element.
    :param schema: additional schema for providing XSD types and elements to the     WSDL document. Can be a :class:`xmlschema.XMLSchema` instance or a file-like     object or a file path or a URL of a resource or a string containing the XSD schema.
    :param cls: class to use for building the schema instance (for default     :class:`xmlschema.XMLSchema10` is used).
    :param validation: the XSD validation mode to use for validating the XML document,     that can be 'strict' (default), 'lax' or 'skip'.
    :param maps: WSDL definitions shared maps.
    :param namespaces: is an optional mapping from namespace prefix to URI.
    :param locations: resource location hints, that can be a dictionary or a     sequence of couples (namespace URI, resource URL).
    :param base_url: the base URL for base :class:`xmlschema.XMLResource` initialization.
    :param allow: the security mode for base :class:`xmlschema.XMLResource` initialization.
    :param defuse: the defuse mode for base :class:`xmlschema.XMLResource` initialization.
    :param timeout: the timeout for base :class:`xmlschema.XMLResource` initialization.
    """
    target_namespace = ''
    soap_binding = False

    def __init__(self, source, schema=None, cls=None, validation='strict', namespaces=None, maps=None, locations=None, base_url=None, allow='all', defuse='remote', timeout=300):
        if maps is not None:
            self.maps = maps
            self.schema = maps.wsdl_document.schema
        else:
            if cls is None:
                cls = XMLSchema10
            if isinstance(schema, XMLSchemaBase):
                cls = schema.__class__
                global_maps = schema.maps
            elif schema is not None:
                global_maps = cls(schema).maps
            else:
                global_maps = None
            self.schema = cls(source=os.path.join(SCHEMAS_DIR, 'WSDL/wsdl.xsd'), global_maps=global_maps)
            self.maps = Wsdl11Maps(self)
        if locations:
            self.locations = NamespaceResourcesMap(locations)
        else:
            self.locations = NamespaceResourcesMap()
        super(Wsdl11Document, self).__init__(source=source, schema=self.schema, validation=validation, namespaces=namespaces, locations=locations, base_url=base_url, allow=allow, defuse=defuse, timeout=timeout)

    @property
    def imports(self):
        """WSDL 1.1 imports of XSD or WSDL additional resources."""
        return self.maps.imports

    @property
    def messages(self):
        """WSDL 1.1 messages."""
        return self.maps.messages

    @property
    def port_types(self):
        """WSDL 1.1 port types."""
        return self.maps.port_types

    @property
    def bindings(self):
        """WSDL 1.1 bindings."""
        return self.maps.bindings

    @property
    def services(self):
        """WSDL 1.1 services."""
        return self.maps.services

    def parse(self, source, lazy=False):
        if lazy:
            raise WsdlParseError('{!r} instance cannot be lazy'.format(self.__class__))
        super(Wsdl11Document, self).parse(source, lazy)
        self.target_namespace = self._root.get('targetNamespace', '')
        self.soap_binding = SOAP_NAMESPACE in self.namespaces.values()
        if self.namespace == XSD_NAMESPACE:
            self.schema.__class__(self, global_maps=self.schema.maps, locations=self.locations)
            return
        if self is self.maps.wsdl_document:
            self.maps.clear()
        self._parse_imports()
        self._parse_types()
        self._parse_messages()
        self._parse_port_types()
        self._parse_bindings()
        self._parse_services()

    def parse_error(self, message):
        if self.validation == 'strict':
            raise WsdlParseError(message)
        elif self.validation == 'lax':
            self.errors.append(WsdlParseError(message))

    def _parse_types(self):
        path = '{}/{}'.format(WSDL_TYPES, XSD_SCHEMA)
        for child in self._root.iterfind(path):
            source = self.subresource(child)
            self.schema.__class__(source, global_maps=self.schema.maps)

    def _parse_messages(self):
        for child in self.iterfind(WSDL_MESSAGE):
            message = WsdlMessage(child, self)
            if message.name in self.maps.messages:
                self.parse_error('duplicated message {!r}'.format(message.prefixed_name))
            else:
                self.maps.messages[message.name] = message

    def _parse_port_types(self):
        for child in self.iterfind(WSDL_PORT_TYPE):
            port_type = WsdlPortType(child, self)
            if port_type.name in self.maps.port_types:
                self.parse_error('duplicated port type {!r}'.format(port_type.prefixed_name))
            else:
                self.maps.port_types[port_type.name] = port_type

    def _parse_bindings(self):
        for child in self.iterfind(WSDL_BINDING):
            binding = WsdlBinding(child, self)
            if binding.name in self.maps.bindings:
                self.parse_error('duplicated binding {!r}'.format(binding.prefixed_name))
            else:
                self.maps.bindings[binding.name] = binding

    def _parse_services(self):
        for child in self.iterfind(WSDL_SERVICE):
            service = WsdlService(child, self)
            if service.name in self.maps.services:
                self.parse_error('duplicated service {!r}'.format(service.prefixed_name))
            else:
                self.maps.services[service.name] = service

    def _parse_imports(self):
        namespace_imports = NamespaceResourcesMap(map(lambda x: (x.get('namespace', ''), x.get('location', '')), filter(lambda x: x.tag == WSDL_IMPORT, self.root)))
        for namespace, locations in namespace_imports.items():
            locations = [url for url in locations if url]
            try:
                locations.extend(self.locations[namespace])
            except KeyError:
                pass
            import_error = None
            for url in locations:
                try:
                    self.import_namespace(namespace, url, self.base_url)
                except (OSError, IOError) as err:
                    if import_error is None:
                        import_error = err
                except SyntaxError as err:
                    msg = 'cannot import namespace %r: %s.' % (namespace, err)
                    self.parse_error(msg)
                except XMLSchemaValueError as err:
                    self.parse_error(err)
                else:
                    break
            else:
                if import_error is not None:
                    msg = 'import of namespace {!r} from {!r} failed: {}.'
                    self.parse_error(msg.format(namespace, locations, str(import_error)))
                self.maps.imports[namespace] = None

    def import_namespace(self, namespace, location, base_url=None):
        if namespace == self.target_namespace:
            msg = "namespace to import must be different from the 'targetNamespace' of the WSDL document"
            raise XMLSchemaValueError(msg)
        elif namespace in self.maps.imports:
            return self.maps.imports[namespace]
        url = fetch_resource(location, base_url or self.base_url)
        wsdl_document = self.__class__(source=url, maps=self.maps, namespaces=self._namespaces, validation=self.validation, base_url=self.base_url, allow=self.allow, defuse=self.defuse, timeout=self.timeout)
        if wsdl_document.target_namespace != namespace:
            msg = 'imported {!r} has an unmatched namespace {!r}'
            self.parse_error(msg.format(wsdl_document, namespace))
        self.maps.imports[namespace] = wsdl_document
        return wsdl_document