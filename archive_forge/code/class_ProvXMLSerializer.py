import datetime
import logging
from lxml import etree
import io
import warnings
import prov
import prov.identifier
from prov.model import DEFAULT_NAMESPACES, sorted_attributes
from prov.constants import *  # NOQA
from prov.serializers import Serializer
class ProvXMLSerializer(Serializer):
    """PROV-XML serializer for :class:`~prov.model.ProvDocument`"""

    def serialize(self, stream, force_types=False, **kwargs):
        """
        Serializes a :class:`~prov.model.ProvDocument` instance to `PROV-XML
        <http://www.w3.org/TR/prov-xml/>`_.

        :param stream: Where to save the output.
        :type force_types: boolean, optional
        :param force_types: Will force xsd:types to be written for most
            attributes mainly PROV-"attributes", e.g. tags not in the
            PROV namespace. Off by default meaning xsd:type attributes will
            only be set for prov:type, prov:location, and prov:value as is
            done in the official PROV-XML specification. Furthermore the
            types will always be set if the Python type requires it. False
            is a good default and it should rarely require changing.
        """
        xml_root = self.serialize_bundle(bundle=self.document, force_types=force_types)
        for bundle in self.document.bundles:
            self.serialize_bundle(bundle=bundle, element=xml_root, force_types=force_types)
        et = etree.ElementTree(xml_root)
        if isinstance(stream, io.TextIOBase):
            stream.write(etree.tostring(et, xml_declaration=True, pretty_print=True).decode('utf-8'))
        else:
            et.write(stream, pretty_print=True, xml_declaration=True, encoding='UTF-8')

    def serialize_bundle(self, bundle, element=None, force_types=False):
        """
        Serializes a bundle or document to PROV XML.

        :param bundle: The bundle or document.
        :param element: The XML element to write to. Will be created if None.
        :type force_types: boolean, optional
        :param force_types: Will force xsd:types to be written for most
            attributes mainly PROV-"attributes", e.g. tags not in the
            PROV namespace. Off by default meaning xsd:type attributes will
            only be set for prov:type, prov:location, and prov:value as is
            done in the official PROV-XML specification. Furthermore the
            types will always be set if the Python type requires it. False
            is a good default and it should rarely require changing.
        """
        nsmap = {ns.prefix: ns.uri for ns in self.document._namespaces.get_registered_namespaces()}
        if self.document._namespaces._default:
            nsmap[None] = self.document._namespaces._default.uri
        for namespace in bundle.namespaces:
            if namespace not in nsmap:
                nsmap[namespace.prefix] = namespace.uri
        for key, value in DEFAULT_NAMESPACES.items():
            uri = value.uri
            if value.prefix == 'xsd':
                uri = uri.rstrip('#')
            nsmap[value.prefix] = uri
        if element is not None:
            xml_bundle_root = etree.SubElement(element, _ns_prov('bundleContent'), nsmap=nsmap)
        else:
            xml_bundle_root = etree.Element(_ns_prov('document'), nsmap=nsmap)
        if bundle.identifier:
            xml_bundle_root.attrib[_ns_prov('id')] = str(bundle.identifier)
        for record in bundle._records:
            rec_type = record.get_type()
            identifier = str(record._identifier) if record._identifier else None
            if identifier:
                attrs = {_ns_prov('id'): identifier}
            else:
                attrs = None
            attributes = list(record.attributes)
            rec_label = self._derive_record_label(rec_type, attributes)
            elem = etree.SubElement(xml_bundle_root, _ns_prov(rec_label), attrs)
            for attr, value in sorted_attributes(rec_type, attributes):
                subelem = etree.SubElement(elem, _ns(attr.namespace.uri, attr.localpart))
                if isinstance(value, prov.model.Literal):
                    if value.datatype not in [None, PROV['InternationalizedString']]:
                        subelem.attrib[_ns_xsi('type')] = '%s:%s' % (value.datatype.namespace.prefix, value.datatype.localpart)
                    if value.langtag is not None:
                        subelem.attrib[_ns_xml('lang')] = value.langtag
                    v = value.value
                elif isinstance(value, prov.model.QualifiedName):
                    if attr not in PROV_ATTRIBUTE_QNAMES:
                        subelem.attrib[_ns_xsi('type')] = 'xsd:QName'
                    v = str(value)
                elif isinstance(value, datetime.datetime):
                    v = value.isoformat()
                else:
                    v = str(value)
                ALWAYS_CHECK = [bool, datetime.datetime, float, int, prov.identifier.Identifier]
                ALWAYS_CHECK = tuple(ALWAYS_CHECK)
                if (force_types or type(value) in ALWAYS_CHECK or attr in [PROV_TYPE, PROV_LOCATION, PROV_VALUE]) and _ns_xsi('type') not in subelem.attrib and (not str(value).startswith('prov:')) and (not (attr in PROV_ATTRIBUTE_QNAMES and v)) and (attr not in [PROV_ATTR_TIME, PROV_LABEL]):
                    xsd_type = None
                    if isinstance(value, bool):
                        xsd_type = XSD_BOOLEAN
                        v = v.lower()
                    elif isinstance(value, str):
                        xsd_type = XSD_STRING
                    elif isinstance(value, float):
                        xsd_type = XSD_DOUBLE
                    elif isinstance(value, int):
                        xsd_type = XSD_INT
                    elif isinstance(value, datetime.datetime):
                        if attr.namespace.prefix != 'prov' or 'time' not in attr.localpart.lower():
                            xsd_type = XSD_DATETIME
                    elif isinstance(value, prov.identifier.Identifier):
                        xsd_type = XSD_ANYURI
                    if xsd_type is not None:
                        subelem.attrib[_ns_xsi('type')] = str(xsd_type)
                if attr in PROV_ATTRIBUTE_QNAMES and v:
                    subelem.attrib[_ns_prov('ref')] = v
                else:
                    subelem.text = v
        return xml_bundle_root

    def deserialize(self, stream, **kwargs):
        """
        Deserialize from `PROV-XML <http://www.w3.org/TR/prov-xml/>`_
        representation to a :class:`~prov.model.ProvDocument` instance.

        :param stream: Input data.
        """
        if isinstance(stream, io.TextIOBase):
            with io.BytesIO() as buf:
                buf.write(stream.read().encode('utf-8'))
                buf.seek(0, 0)
                xml_doc = etree.parse(buf).getroot()
        else:
            xml_doc = etree.parse(stream).getroot()
        for c in xml_doc.xpath('//comment()'):
            p = c.getparent()
            p.remove(c)
        document = prov.model.ProvDocument()
        self.deserialize_subtree(xml_doc, document)
        return document

    def deserialize_subtree(self, xml_doc, bundle):
        """
        Deserialize an etree element containing a PROV document or a bundle
        and write it to the provided internal object.

        :param xml_doc: An etree element containing the information to read.
        :param bundle: The bundle object to write to.
        """
        for element in xml_doc:
            qname = etree.QName(element)
            if qname.namespace != DEFAULT_NAMESPACES['prov'].uri:
                raise ProvXMLException('Non PROV element discovered in document or bundle.')
            if qname.localname == 'other':
                warnings.warn('Document contains non-PROV information in <prov:other>. It will be ignored in this package.', UserWarning)
                continue
            id_tag = _ns_prov('id')
            rec_id = element.attrib[id_tag] if id_tag in element.attrib else None
            if rec_id is not None:
                rec_id = xml_qname_to_QualifiedName(element, rec_id)
            if qname.localname == 'bundleContent':
                b = bundle.bundle(identifier=rec_id)
                self.deserialize_subtree(element, b)
                continue
            attributes = _extract_attributes(element)
            q_prov_name = FULL_PROV_RECORD_IDS_MAP[qname.localname]
            rec_type = PROV_BASE_CLS[q_prov_name]
            if _ns_xsi('type') in element.attrib:
                value = xml_qname_to_QualifiedName(element, element.attrib[_ns_xsi('type')])
                attributes.append((PROV['type'], value))
            rec = bundle.new_record(rec_type, rec_id, attributes)
            if rec_type != q_prov_name:
                rec.add_asserted_type(q_prov_name)
        return bundle

    def _derive_record_label(self, rec_type, attributes):
        """
        Helper function trying to derive the record label taking care of
        subtypes and what not. It will also remove the type declaration for
        the attributes if it was used to specialize the type.

        :param rec_type: The type of records.
        :param attributes: The attributes of the record.
        """
        rec_label = FULL_NAMES_MAP[rec_type]
        for key, value in list(attributes):
            if key != PROV_TYPE:
                continue
            if isinstance(value, prov.model.Literal):
                value = value.value
            if value in PROV_BASE_CLS and PROV_BASE_CLS[value] != value:
                attributes.remove((key, value))
                rec_label = FULL_NAMES_MAP[value]
                break
        return rec_label