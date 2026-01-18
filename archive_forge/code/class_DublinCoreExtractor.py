import re
from w3lib.html import strip_html5_whitespace
from extruct.utils import parse_html
class DublinCoreExtractor:
    """DublinCore extractor following extruct API."""

    def extract(self, htmlstring, base_url=None, encoding='UTF-8'):
        tree = parse_html(htmlstring, encoding=encoding)
        return list(self.extract_items(tree, base_url=base_url))

    def extract_items(self, document, base_url=None):
        elements = []
        terms = []

        def attrib_to_dict(attribs):
            return dict(attribs.items())

        def populate_results(node, main_attrib):
            node_attrib = node.attrib
            if main_attrib not in node_attrib:
                return
            name = node.attrib[main_attrib]
            lower_name = get_lower_attrib(name)
            if lower_name in _DC_ELEMENTS:
                node.attrib.update({'URI': _DC_ELEMENTS[lower_name]})
                elements.append(attrib_to_dict(node.attrib))
            elif lower_name in _DC_TERMS:
                node.attrib.update({'URI': _DC_TERMS[lower_name]})
                terms.append(attrib_to_dict(node.attrib))
        namespaces_nodes = document.xpath('//link[contains(@rel,"schema")]')
        namespaces = {}
        for i in namespaces_nodes:
            url = strip_html5_whitespace(i.attrib['href'])
            if url in _URL_NAMESPACES:
                namespaces.update({re.sub('schema\\.', '', i.attrib['rel']): url})
        list_meta_node = document.xpath('//meta')
        for meta_node in list_meta_node:
            populate_results(meta_node, 'name')
        list_link_node = document.xpath('//link')
        for link_node in list_link_node:
            populate_results(link_node, 'rel')
        yield {'namespaces': namespaces, 'elements': elements, 'terms': terms}