import lxml
import os
import os.path as op
import sys
import re
import shutil
import tempfile
import zipfile
import codecs
from fnmatch import fnmatch
from itertools import islice
from lxml import etree
from pathlib import Path
from .uriutil import join_uri, translate_uri, uri_segment
from .uriutil import uri_last, uri_nextlast
from .uriutil import uri_parent, uri_grandparent
from .uriutil import uri_shape
from .uriutil import file_path
from .jsonutil import JsonTable, get_selection
from .pathutil import find_files, ensure_dir_exists
from .attributes import EAttrs
from .search import rpn_contraints, query_from_xml
from .errors import is_xnat_error, parse_put_error_message
from .errors import DataError, ProgrammingError, catch_error
from .provenance import Provenance
from .pipelines import Pipelines
from . import schema
from . import httputil
from . import downloadutils
from . import derivatives
import types
import pkgutil
import inspect
from urllib.parse import quote, unquote
def add_custom_variables(self, custom_variables, allow_data_deletion=False):
    """Adds a custom variable to a specified group

        Parameters
        ----------

        custom_variables: a dictionary
        allow_data_deletion : a boolean

        Examples
        --------

        >>> variables = {'Subjects' : {'newgroup' : {'foo' : 'string',
            'bar': 'int'}}}
        >>> project.add_custom_variables(variables)

        """
    tree = lxml.etree.fromstring(self.get())
    update = False
    for protocol, value in custom_variables.items():
        try:
            protocol_element = tree.xpath("//xnat:studyProtocol[@name='%s']" % protocol, namespaces=tree.nsmap).pop()
        except IndexError:
            raise ValueError('Protocol %s not in current schema' % protocol)
        try:
            definitions_element = protocol_element.xpath('xnat:definitions', namespaces=tree.nsmap).pop()
        except IndexError:
            update = True
            definitions_element = lxml.etree.Element(lxml.etree.QName(tree.nsmap['xnat'], 'definitions'), nsmap=tree.nsmap)
            protocol_element.append(definitions_element)
        for group, fields in value.items():
            try:
                group_element = definitions_element.xpath("xnat:definition[@ID='%s']" % group, namespaces=tree.nsmap).pop()
                fields_element = group_element.xpath('xnat:fields', namespaces=tree.nsmap).pop()
            except IndexError:
                update = True
                group_element = lxml.etree.Element(lxml.etree.QName(tree.nsmap['xnat'], 'definition'), nsmap=tree.nsmap)
                group_element.set('ID', group)
                group_element.set('data-type', protocol_element.get('data-type'))
                group_element.set('description', '')
                group_element.set('project-specific', '1')
                definitions_element.append(group_element)
                fields_element = lxml.etree.Element(lxml.etree.QName(tree.nsmap['xnat'], 'fields'), nsmap=tree.nsmap)
                group_element.append(fields_element)
            for field, datatype in fields.items():
                try:
                    field_element = fields_element.xpath("xnat:field[@name='%s']" % field, namespaces=tree.nsmap).pop()
                except IndexError:
                    field_element = lxml.etree.Element(lxml.etree.QName(tree.nsmap['xnat'], 'field'), nsmap=tree.nsmap)
                    field_element.set('name', field)
                    field_element.set('datatype', datatype)
                    field_element.set('type', 'custom')
                    field_element.set('required', '0')
                    field_element.set('xmlPath', 'xnat:%s/fields/field[name=%s]/field' % (protocol_element.get('data-type').split(':')[-1], field))
                    fields_element.append(field_element)
                    update = True
    if update:
        body, content_type = httputil.file_message(lxml.etree.tostring(tree).decode('utf-8'), 'text/xml', 'cust.xml', 'cust.xml')
        uri = self._uri
        if allow_data_deletion:
            uri = self._uri + '?allowDataDeletion=true'
        self._intf._exec(uri, method='PUT', body=str(body), headers={'content-type': content_type})