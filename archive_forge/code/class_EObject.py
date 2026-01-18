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
class EObject(object):
    """ Generic Object for an element URI.
    """

    def __init__(self, uri, interface):
        """
            Parameters
            ----------
            uri: string
                URI for an element resource.
                e.g. /REST/projects/my_project

            interface: :class:`Interface`
                Main interface reference.
        """
        self._uri = quote(translate_uri(uri))
        self._urn = unquote(uri_last(self._uri))
        self._urt = uri_nextlast(self._uri)
        self._intf = interface
        self.attrs = EAttrs(self)
        functions = __find_all_functions__(derivatives)
        for m, mod_functions in functions.items():
            is_resource = False
            if hasattr(m, 'XNAT_RESOURCE_NAME') and self._urn == m.XNAT_RESOURCE_NAME or (hasattr(m, 'XNAT_RESOURCE_NAMES') and self._urn in m.XNAT_RESOURCE_NAMES):
                is_resource = True
            if is_resource:
                for f in mod_functions:
                    setattr(self, f.__name__, types.MethodType(f, self))

    def __getstate__(self):
        return {'uri': self._uri, 'interface': self._intf}

    def __setstate__(self, dict):
        self.__init__(dict['uri'], dict['interface'])

    def __repr__(self):
        return '<%s Object> %s' % (self.__class__.__name__, unquote(uri_last(self._uri)))

    def _getcell(self, col):
        """ Gets a single property of the element resource.
        """
        return self._getcells([col])

    def _getcells(self, cols):
        """ Gets multiple properties of the element resource.
        """
        p_uri = uri_parent(self._uri)
        id_head = schema.json[self._urt][0]
        lbl_head = schema.json[self._urt][1]
        filters = {}
        columns = set([col for col in cols if col not in schema.json[self._urt] or col != 'URI'] + schema.json[self._urt])
        get_id = p_uri + '?format=json&columns=%s' % ','.join(columns)
        for pattern in self._intf._struct.keys():
            if fnmatch(uri_segment(self._uri.split(self._intf._get_entry_point(), 1)[1], -2), pattern):
                reg_pat = self._intf._struct[pattern]
                filters.setdefault('xsiType', set()).add(reg_pat)
        if filters:
            get_id += '&' + '&'.join(('%s=%s' % (item[0], item[1]) if isinstance(item[1], str) else '%s=%s' % (item[0], ','.join([val for val in item[1]])) for item in filters.items()))
        for res in self._intf._get_json(get_id):
            if self._urn in [res.get(id_head), res.get(lbl_head)]:
                if len(cols) == 1:
                    return res.get(cols[0])
                else:
                    return get_selection(res, cols)[0]

    def exists(self, consistent=False):
        """ Test whether an element resource exists.
        """
        try:
            return self.id() is not None
        except Exception as e:
            if DEBUG:
                print(e)
            return False

    def id(self):
        """ Returns the element resource id.
        """
        return self._getcell(schema.json[self._urt][0])

    def label(self):
        """ Returns the element resource label.
        """
        return self._getcell(schema.json[self._urt][1])

    def datatype(self):
        """ Returns the type defined in the XNAT schema for this element
        resource.

            +----------------+-----------------------+
            | EObject        | possible xsi types    |
            +================+=======================+
            | Project        | xnat:projectData      |
            +----------------+-----------------------+
            | Subject        | xnat:subjectData      |
            +----------------+-----------------------+
            | Experiment     | xnat:mrSessionData    |
            |                | xnat:petSessionData   |
            +----------------+-----------------------+
        """
        return self._getcell('xsiType')

    def create(self, **params):
        """ Creates the element if it does not exists.
            Any non-existing ancestor will be created as well.

            .. warning::
                An element resource both have an ID and a label that
                can be used to access it. At the moment, XNAT REST API
                defines the label when creating an element, but not
                the ID, which is generated. It means that the `name`
                given to a resource may not appear when listing the
                resources because the IDs will appear, not the labels.

            .. note::
               To set up additional variables for the element at its
               creation it is possible to use shortcuts defined in the
               XNAT REST documentation or xpath in the schema:

                   - `element.create(ID='theid')`
                   - `subject.create(**{'xnat:subjectData/ID':'theid'})`


            Parameters
            ----------
            params: keywords

                - Specify the datatype of the element resource and of
                  any ancestor that may need to be created. The
                  keywords correspond to the levels in the REST
                  hierarchy, see Interface.inspect.architecture()

                - If an element is created with no specified type:

                      - if its name matches a naming convention, this type
                        will be used
                      - else a default type is defined in the schema module

                - To give the ID the same value as the label use
                  use_label=True e.g element.create(use_label=True)

            Examples
            --------

                >>> interface.select('/project/PROJECT/subject'
                                     '/SUBJECT/experiment/EXP/scan/SCAN'
                            ).create(experiments='xnat:mrSessionData',
                                     scans='xnat:mrScanData'
                                     )

            See Also
            --------
            :func:`EObject.id`
            :func:`EObject.label`
            :func:`EObject.datatype`
        """
        if 'xml' in params and op.exists(params.get('xml')):
            f = codecs.open(params.get('xml'))
            doc = f.read()
            f.close()
            try:
                doc_tree = etree.fromstring(doc)
                doc_tree.xpath('//*')[0].set('label', uri_last(self._uri))
                doc = etree.tostring(doc_tree)
            except Exception:
                pass
            body, content_type = httputil.file_message(doc.decode(), 'text/xml', 'data.xml', 'data.xml')
            _uri = self._uri
            if 'allowDataDeletion' in params and params.get('allowDataDeletion') is False:
                _uri += '?allowDataDeletion=false'
            else:
                _uri += '?allowDataDeletion=true'
            self._intf._exec(_uri, method='PUT', body=body, headers={'content-type': content_type})
            return self
        datatype = params.get(uri_nextlast(self._uri))
        struct = self._intf._struct
        if datatype is None:
            for uri_pattern in struct.keys():
                if fnmatch(self._uri.split(self._intf._get_entry_point(), 1)[1], uri_pattern):
                    datatype = struct.get(uri_pattern)
                    break
            else:
                datatype = schema.default_datatypes.get(uri_nextlast(self._uri))
        if datatype is None:
            create_uri = self._uri
        else:
            local_params = [param for param in params if param not in schema.resources_types + ['use_label'] and (param.startswith(datatype) or '/' not in param)]
            create_uri = '%s?xsiType=%s' % (self._uri, datatype)
            if 'ID' not in local_params and '%s/ID' % datatype not in local_params and params.get('use_label'):
                create_uri += '&%s/ID=%s' % (datatype, uri_last(self._uri))
            if local_params:
                create_uri += '&' + '&'.join(('%s=%s' % (key, params.get(key)) for key in local_params))
            for key in local_params:
                del params[key]
        parent_element = self._intf.select(uri_grandparent(self._uri))
        if not uri_nextlast(self._uri) == 'projects' and (not parent_element.exists()):
            parent_datatype = params.get(uri_nextlast(parent_element._uri))
            if DEBUG:
                print('CREATE', parent_element, parent_datatype)
            parent_element.create(**params)
        if DEBUG:
            print('PUT', create_uri)
        if 'params' in params and 'event_reason' in params['params']:
            if DEBUG:
                print('Found event_reason')
            output = self._intf._exec(create_uri, 'PUT', **params)
        else:
            if DEBUG:
                print('event_reason not found')
            output = self._intf._exec(create_uri, 'PUT')
        if is_xnat_error(output):
            paths = []
            for datatype_name, element_name in parse_put_error_message(output):
                path = self._intf.inspect.schemas.look_for(element_name, datatype_name)
                paths.extend(path)
                if DEBUG:
                    print(path, 'is required')
            return paths
        return self
    insert = create

    def delete(self, delete_files=True):
        """ Deletes an element resource.

            Parameters
            ----------
            delete_files: boolean
                Tells if files attached to the element resources are
                removed as well from the server filesystem.
        """
        delete_uri = self._uri if not delete_files else self._uri + '?removeFiles=true'
        out = self._intf._exec(delete_uri, 'DELETE')
        if is_xnat_error(out):
            catch_error(out)

    def get(self):
        """ Retrieves the XML document corresponding to this element.
        """
        return self._intf._exec(self._uri + '?format=xml', 'GET')

    def xpath(self, xpath):
        root = etree.fromstring(self.get())
        return root.xpath(xpath, namespaces=root.nsmap)

    def namespaces(self):
        pass

    def parent(self):
        uri = uri_grandparent(self._uri)
        klass = uri_nextlast(uri).title().rsplit('s', 1)[0]
        if klass:
            Klass = globals()[klass]
            return Klass(uri, self._intf)
        else:
            return None

    def children(self, show_names=True):
        """ Returns the children levels of this element.

            Parameters
            ----------
            show_name: boolean
                If True returns a list of strings. If False returns a
                collection object referencing all child objects of
                this elements.

            Examples
            --------
            >>> subject_object.children()
            ['experiments', 'resources']
            >>> subject_object.children(False)
            <Collection Object> 170976556
        """
        children = schema.resources_tree.get(uri_nextlast(self._uri))
        if show_names:
            return children
        return CObject([getattr(self, child)() for child in children], self._intf)

    def tag(self, name):
        """ Tag the element.
        """
        tag = self._intf.manage.tags.get(name)
        if not tag.exists():
            tag.create()
        tag.reference(self._uri)
        return tag

    def untag(self, name):
        """ Remove a tag for the element.
        """
        tag = self._intf.manage.tags.get(name)
        tag.dereference(self._uri)
        if not tag.references().get():
            tag.delete()