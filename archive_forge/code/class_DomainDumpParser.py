from __future__ import print_function
from boto.sdb.queryresultset import SelectResultSet
from boto.compat import six
import sys
from xml.sax.handler import ContentHandler
from threading import Thread
class DomainDumpParser(ContentHandler):
    """
    SAX parser for a domain that has been dumped
    """

    def __init__(self, domain):
        self.uploader = UploaderThread(domain)
        self.item_id = None
        self.attrs = {}
        self.attribute = None
        self.value = ''
        self.domain = domain

    def startElement(self, name, attrs):
        if name == 'Item':
            self.item_id = attrs['id']
            self.attrs = {}
        elif name == 'attribute':
            self.attribute = attrs['id']
        elif name == 'value':
            self.value = ''

    def characters(self, ch):
        self.value += ch

    def endElement(self, name):
        if name == 'value':
            if self.value and self.attribute:
                value = self.value.strip()
                attr_name = self.attribute.strip()
                if attr_name in self.attrs:
                    self.attrs[attr_name].append(value)
                else:
                    self.attrs[attr_name] = [value]
        elif name == 'Item':
            self.uploader.items[self.item_id] = self.attrs
            if len(self.uploader.items) >= 20:
                self.uploader.start()
                self.uploader = UploaderThread(self.domain)
        elif name == 'Domain':
            self.uploader.start()