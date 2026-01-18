import zipfile
import six
import logging
import uuid
import warnings
import posixpath as zip_path
import os.path
from collections import OrderedDict
from lxml import etree
import ebooklib
from ebooklib.utils import parse_string, parse_html_string, guess_type, get_pages_for_items
def _write_opf_metadata(self, root):
    nsmap = {'dc': NAMESPACES['DC'], 'opf': NAMESPACES['OPF']}
    nsmap.update(self.book.namespaces)
    metadata = etree.SubElement(root, 'metadata', nsmap=nsmap)
    el = etree.SubElement(metadata, 'meta', {'property': 'dcterms:modified'})
    if 'mtime' in self.options:
        mtime = self.options['mtime']
    else:
        import datetime
        mtime = datetime.datetime.now()
    el.text = mtime.strftime('%Y-%m-%dT%H:%M:%SZ')
    for ns_name, values in six.iteritems(self.book.metadata):
        if ns_name == NAMESPACES['OPF']:
            for values in values.values():
                for v in values:
                    if 'property' in v[1] and v[1]['property'] == 'dcterms:modified':
                        continue
                    try:
                        el = etree.SubElement(metadata, 'meta', v[1])
                        if v[0]:
                            el.text = v[0]
                    except ValueError:
                        logging.error('Could not create metadata.')
        else:
            for name, values in six.iteritems(values):
                for v in values:
                    try:
                        if ns_name:
                            el = etree.SubElement(metadata, '{%s}%s' % (ns_name, name), v[1])
                        else:
                            el = etree.SubElement(metadata, '%s' % name, v[1])
                        el.text = v[0]
                    except ValueError:
                        logging.error('Could not create metadata "{}".'.format(name))