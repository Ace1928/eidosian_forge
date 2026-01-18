from suds import *
from suds.xsd import *
from suds.xsd.depsort import dependency_sort
from suds.xsd.sxbuiltin import *
from suds.xsd.sxbase import SchemaObject
from suds.xsd.sxbasic import Factory as BasicFactory
from suds.xsd.sxbuiltin import Factory as BuiltinFactory
from suds.sax import splitPrefix, Namespace
from suds.sax.element import Element
from logging import getLogger
class SchemaCollection(UnicodeMixin):
    """
    A collection of schema objects.

    This class is needed because a WSDL may contain more then one <schema/>
    node.

    @ivar wsdl: A WSDL object.
    @type wsdl: L{suds.wsdl.Definitions}
    @ivar children: A list contained schemas.
    @type children: [L{Schema},...]
    @ivar namespaces: A dictionary of contained schemas by namespace.
    @type namespaces: {str: L{Schema}}

    """

    def __init__(self, wsdl):
        """
        @param wsdl: A WSDL object.
        @type wsdl: L{suds.wsdl.Definitions}

        """
        self.wsdl = wsdl
        self.children = []
        self.namespaces = {}

    def add(self, schema):
        """
        Add a schema node to the collection. Schema(s) within the same target
        namespace are consolidated.

        @param schema: A schema object.
        @type schema: (L{Schema})

        """
        key = schema.tns[1]
        existing = self.namespaces.get(key)
        if existing is None:
            self.children.append(schema)
            self.namespaces[key] = schema
        else:
            existing.root.children += schema.root.children
            existing.root.nsprefixes.update(schema.root.nsprefixes)

    def load(self, options, loaded_schemata):
        """
        Load schema objects for the root nodes.
            - de-reference schemas
            - merge schemas

        @param options: An options dictionary.
        @type options: L{options.Options}
        @param loaded_schemata: Already loaded schemata cache (URL --> Schema).
        @type loaded_schemata: dict
        @return: The merged schema.
        @rtype: L{Schema}

        """
        if options.autoblend:
            self.autoblend()
        for child in self.children:
            child.build()
        for child in self.children:
            child.open_imports(options, loaded_schemata)
        for child in self.children:
            child.dereference()
        log.debug('loaded:\n%s', self)
        merged = self.merge()
        log.debug('MERGED:\n%s', merged)
        return merged

    def autoblend(self):
        """
        Ensure that all schemas within the collection import each other which
        has a blending effect.

        @return: self
        @rtype: L{SchemaCollection}

        """
        namespaces = list(self.namespaces.keys())
        for s in self.children:
            for ns in namespaces:
                tns = s.root.get('targetNamespace')
                if tns == ns:
                    continue
                for imp in s.root.getChildren('import'):
                    if imp.get('namespace') == ns:
                        continue
                imp = Element('import', ns=Namespace.xsdns)
                imp.set('namespace', ns)
                s.root.append(imp)
        return self

    def locate(self, ns):
        """
        Find a schema by namespace. Only the URI portion of the namespace is
        compared to each schema's I{targetNamespace}.

        @param ns: A namespace.
        @type ns: (prefix, URI)
        @return: The schema matching the namespace, else None.
        @rtype: L{Schema}

        """
        return self.namespaces.get(ns[1])

    def merge(self):
        """
        Merge contained schemas into one.

        @return: The merged schema.
        @rtype: L{Schema}

        """
        if self.children:
            schema = self.children[0]
            for s in self.children[1:]:
                schema.merge(s)
            return schema

    def __len__(self):
        return len(self.children)

    def __unicode__(self):
        result = ['\nschema collection']
        for s in self.children:
            result.append(s.str(1))
        return '\n'.join(result)