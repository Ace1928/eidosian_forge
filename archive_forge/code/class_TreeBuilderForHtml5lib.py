import warnings
import re
from bs4.builder import (
from bs4.element import (
import html5lib
from html5lib.constants import (
from bs4.element import (
class TreeBuilderForHtml5lib(treebuilder_base.TreeBuilder):

    def __init__(self, namespaceHTMLElements, soup=None, store_line_numbers=True, **kwargs):
        if soup:
            self.soup = soup
        else:
            from bs4 import BeautifulSoup
            self.soup = BeautifulSoup('', 'html.parser', store_line_numbers=store_line_numbers, **kwargs)
        super(TreeBuilderForHtml5lib, self).__init__(namespaceHTMLElements)
        self.parser = None
        self.store_line_numbers = store_line_numbers

    def documentClass(self):
        self.soup.reset()
        return Element(self.soup, self.soup, None)

    def insertDoctype(self, token):
        name = token['name']
        publicId = token['publicId']
        systemId = token['systemId']
        doctype = Doctype.for_name_and_ids(name, publicId, systemId)
        self.soup.object_was_parsed(doctype)

    def elementClass(self, name, namespace):
        kwargs = {}
        if self.parser and self.store_line_numbers:
            sourceline, sourcepos = self.parser.tokenizer.stream.position()
            kwargs['sourceline'] = sourceline
            kwargs['sourcepos'] = sourcepos - 1
        tag = self.soup.new_tag(name, namespace, **kwargs)
        return Element(tag, self.soup, namespace)

    def commentClass(self, data):
        return TextNode(Comment(data), self.soup)

    def fragmentClass(self):
        from bs4 import BeautifulSoup
        self.soup = BeautifulSoup('', 'html.parser')
        self.soup.name = '[document_fragment]'
        return Element(self.soup, self.soup, None)

    def appendChild(self, node):
        self.soup.append(node.element)

    def getDocument(self):
        return self.soup

    def getFragment(self):
        return treebuilder_base.TreeBuilder.getFragment(self).element

    def testSerializer(self, element):
        from bs4 import BeautifulSoup
        rv = []
        doctype_re = re.compile('^(.*?)(?: PUBLIC "(.*?)"(?: "(.*?)")?| SYSTEM "(.*?)")?$')

        def serializeElement(element, indent=0):
            if isinstance(element, BeautifulSoup):
                pass
            if isinstance(element, Doctype):
                m = doctype_re.match(element)
                if m:
                    name = m.group(1)
                    if m.lastindex > 1:
                        publicId = m.group(2) or ''
                        systemId = m.group(3) or m.group(4) or ''
                        rv.append('|%s<!DOCTYPE %s "%s" "%s">' % (' ' * indent, name, publicId, systemId))
                    else:
                        rv.append('|%s<!DOCTYPE %s>' % (' ' * indent, name))
                else:
                    rv.append('|%s<!DOCTYPE >' % (' ' * indent,))
            elif isinstance(element, Comment):
                rv.append('|%s<!-- %s -->' % (' ' * indent, element))
            elif isinstance(element, NavigableString):
                rv.append('|%s"%s"' % (' ' * indent, element))
            else:
                if element.namespace:
                    name = '%s %s' % (prefixes[element.namespace], element.name)
                else:
                    name = element.name
                rv.append('|%s<%s>' % (' ' * indent, name))
                if element.attrs:
                    attributes = []
                    for name, value in list(element.attrs.items()):
                        if isinstance(name, NamespacedAttribute):
                            name = '%s %s' % (prefixes[name.namespace], name.name)
                        if isinstance(value, list):
                            value = ' '.join(value)
                        attributes.append((name, value))
                    for name, value in sorted(attributes):
                        rv.append('|%s%s="%s"' % (' ' * (indent + 2), name, value))
                indent += 2
                for child in element.children:
                    serializeElement(child, indent)
        serializeElement(element, 0)
        return '\n'.join(rv)