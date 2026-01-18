import boto
from boto.utils import find_class, Password
from boto.sdb.db.key import Key
from boto.sdb.db.model import Model
from boto.compat import six, encodebytes
from datetime import datetime
from xml.dom.minidom import getDOMImplementation, parse, parseString, Node
class XMLManager(object):

    def __init__(self, cls, db_name, db_user, db_passwd, db_host, db_port, db_table, ddl_dir, enable_ssl):
        self.cls = cls
        if not db_name:
            db_name = cls.__name__.lower()
        self.db_name = db_name
        self.db_user = db_user
        self.db_passwd = db_passwd
        self.db_host = db_host
        self.db_port = db_port
        self.db_table = db_table
        self.ddl_dir = ddl_dir
        self.s3 = None
        self.converter = XMLConverter(self)
        self.impl = getDOMImplementation()
        self.doc = self.impl.createDocument(None, 'objects', None)
        self.connection = None
        self.enable_ssl = enable_ssl
        self.auth_header = None
        if self.db_user:
            base64string = encodebytes('%s:%s' % (self.db_user, self.db_passwd))[:-1]
            authheader = 'Basic %s' % base64string
            self.auth_header = authheader

    def _connect(self):
        if self.db_host:
            if self.enable_ssl:
                from httplib import HTTPSConnection as Connection
            else:
                from httplib import HTTPConnection as Connection
            self.connection = Connection(self.db_host, self.db_port)

    def _make_request(self, method, url, post_data=None, body=None):
        """
        Make a request on this connection
        """
        if not self.connection:
            self._connect()
        try:
            self.connection.close()
        except:
            pass
        self.connection.connect()
        headers = {}
        if self.auth_header:
            headers['Authorization'] = self.auth_header
        self.connection.request(method, url, body, headers)
        resp = self.connection.getresponse()
        return resp

    def new_doc(self):
        return self.impl.createDocument(None, 'objects', None)

    def _object_lister(self, cls, doc):
        for obj_node in doc.getElementsByTagName('object'):
            if not cls:
                class_name = obj_node.getAttribute('class')
                cls = find_class(class_name)
            id = obj_node.getAttribute('id')
            obj = cls(id)
            for prop_node in obj_node.getElementsByTagName('property'):
                prop_name = prop_node.getAttribute('name')
                prop = obj.find_property(prop_name)
                if prop:
                    if hasattr(prop, 'item_type'):
                        value = self.get_list(prop_node, prop.item_type)
                    else:
                        value = self.decode_value(prop, prop_node)
                        value = prop.make_value_from_datastore(value)
                    setattr(obj, prop.name, value)
            yield obj

    def reset(self):
        self._connect()

    def get_doc(self):
        return self.doc

    def encode_value(self, prop, value):
        return self.converter.encode_prop(prop, value)

    def decode_value(self, prop, value):
        return self.converter.decode_prop(prop, value)

    def get_s3_connection(self):
        if not self.s3:
            self.s3 = boto.connect_s3(self.aws_access_key_id, self.aws_secret_access_key)
        return self.s3

    def get_list(self, prop_node, item_type):
        values = []
        try:
            items_node = prop_node.getElementsByTagName('items')[0]
        except:
            return []
        for item_node in items_node.getElementsByTagName('item'):
            value = self.converter.decode(item_type, item_node)
            values.append(value)
        return values

    def get_object_from_doc(self, cls, id, doc):
        obj_node = doc.getElementsByTagName('object')[0]
        if not cls:
            class_name = obj_node.getAttribute('class')
            cls = find_class(class_name)
        if not id:
            id = obj_node.getAttribute('id')
        obj = cls(id)
        for prop_node in obj_node.getElementsByTagName('property'):
            prop_name = prop_node.getAttribute('name')
            prop = obj.find_property(prop_name)
            value = self.decode_value(prop, prop_node)
            value = prop.make_value_from_datastore(value)
            if value is not None:
                try:
                    setattr(obj, prop.name, value)
                except:
                    pass
        return obj

    def get_props_from_doc(self, cls, id, doc):
        """
        Pull out the properties from this document
        Returns the class, the properties in a hash, and the id if provided as a tuple
        :return: (cls, props, id)
        """
        obj_node = doc.getElementsByTagName('object')[0]
        if not cls:
            class_name = obj_node.getAttribute('class')
            cls = find_class(class_name)
        if not id:
            id = obj_node.getAttribute('id')
        props = {}
        for prop_node in obj_node.getElementsByTagName('property'):
            prop_name = prop_node.getAttribute('name')
            prop = cls.find_property(prop_name)
            value = self.decode_value(prop, prop_node)
            value = prop.make_value_from_datastore(value)
            if value is not None:
                props[prop.name] = value
        return (cls, props, id)

    def get_object(self, cls, id):
        if not self.connection:
            self._connect()
        if not self.connection:
            raise NotImplementedError("Can't query without a database connection")
        url = '/%s/%s' % (self.db_name, id)
        resp = self._make_request('GET', url)
        if resp.status == 200:
            doc = parse(resp)
        else:
            raise Exception('Error: %s' % resp.status)
        return self.get_object_from_doc(cls, id, doc)

    def query(self, cls, filters, limit=None, order_by=None):
        if not self.connection:
            self._connect()
        if not self.connection:
            raise NotImplementedError("Can't query without a database connection")
        from urllib import urlencode
        query = str(self._build_query(cls, filters, limit, order_by))
        if query:
            url = '/%s?%s' % (self.db_name, urlencode({'query': query}))
        else:
            url = '/%s' % self.db_name
        resp = self._make_request('GET', url)
        if resp.status == 200:
            doc = parse(resp)
        else:
            raise Exception('Error: %s' % resp.status)
        return self._object_lister(cls, doc)

    def _build_query(self, cls, filters, limit, order_by):
        import types
        if len(filters) > 4:
            raise Exception('Too many filters, max is 4')
        parts = []
        properties = cls.properties(hidden=False)
        for filter, value in filters:
            name, op = filter.strip().split()
            found = False
            for property in properties:
                if property.name == name:
                    found = True
                    if types.TypeType(value) == list:
                        filter_parts = []
                        for val in value:
                            val = self.encode_value(property, val)
                            filter_parts.append("'%s' %s '%s'" % (name, op, val))
                        parts.append('[%s]' % ' OR '.join(filter_parts))
                    else:
                        value = self.encode_value(property, value)
                        parts.append("['%s' %s '%s']" % (name, op, value))
            if not found:
                raise Exception('%s is not a valid field' % name)
        if order_by:
            if order_by.startswith('-'):
                key = order_by[1:]
                type = 'desc'
            else:
                key = order_by
                type = 'asc'
            parts.append("['%s' starts-with ''] sort '%s' %s" % (key, key, type))
        return ' intersection '.join(parts)

    def query_gql(self, query_string, *args, **kwds):
        raise NotImplementedError('GQL queries not supported in XML')

    def save_list(self, doc, items, prop_node):
        items_node = doc.createElement('items')
        prop_node.appendChild(items_node)
        for item in items:
            item_node = doc.createElement('item')
            items_node.appendChild(item_node)
            if isinstance(item, Node):
                item_node.appendChild(item)
            else:
                text_node = doc.createTextNode(item)
                item_node.appendChild(text_node)

    def save_object(self, obj, expected_value=None):
        """
        Marshal the object and do a PUT
        """
        doc = self.marshal_object(obj)
        if obj.id:
            url = '/%s/%s' % (self.db_name, obj.id)
        else:
            url = '/%s' % self.db_name
        resp = self._make_request('PUT', url, body=doc.toxml())
        new_obj = self.get_object_from_doc(obj.__class__, None, parse(resp))
        obj.id = new_obj.id
        for prop in obj.properties():
            try:
                propname = prop.name
            except AttributeError:
                propname = None
            if propname:
                value = getattr(new_obj, prop.name)
                if value:
                    setattr(obj, prop.name, value)
        return obj

    def marshal_object(self, obj, doc=None):
        if not doc:
            doc = self.new_doc()
        if not doc:
            doc = self.doc
        obj_node = doc.createElement('object')
        if obj.id:
            obj_node.setAttribute('id', obj.id)
        obj_node.setAttribute('class', '%s.%s' % (obj.__class__.__module__, obj.__class__.__name__))
        root = doc.documentElement
        root.appendChild(obj_node)
        for property in obj.properties(hidden=False):
            prop_node = doc.createElement('property')
            prop_node.setAttribute('name', property.name)
            prop_node.setAttribute('type', property.type_name)
            value = property.get_value_for_datastore(obj)
            if value is not None:
                value = self.encode_value(property, value)
                if isinstance(value, list):
                    self.save_list(doc, value, prop_node)
                elif isinstance(value, Node):
                    prop_node.appendChild(value)
                else:
                    text_node = doc.createTextNode(six.text_type(value).encode('ascii', 'ignore'))
                    prop_node.appendChild(text_node)
            obj_node.appendChild(prop_node)
        return doc

    def unmarshal_object(self, fp, cls=None, id=None):
        if isinstance(fp, six.string_types):
            doc = parseString(fp)
        else:
            doc = parse(fp)
        return self.get_object_from_doc(cls, id, doc)

    def unmarshal_props(self, fp, cls=None, id=None):
        """
        Same as unmarshalling an object, except it returns
        from "get_props_from_doc"
        """
        if isinstance(fp, six.string_types):
            doc = parseString(fp)
        else:
            doc = parse(fp)
        return self.get_props_from_doc(cls, id, doc)

    def delete_object(self, obj):
        url = '/%s/%s' % (self.db_name, obj.id)
        return self._make_request('DELETE', url)

    def set_key_value(self, obj, name, value):
        self.domain.put_attributes(obj.id, {name: value}, replace=True)

    def delete_key_value(self, obj, name):
        self.domain.delete_attributes(obj.id, name)

    def get_key_value(self, obj, name):
        a = self.domain.get_attributes(obj.id, name)
        if name in a:
            return a[name]
        else:
            return None

    def get_raw_item(self, obj):
        return self.domain.get_item(obj.id)

    def set_property(self, prop, obj, name, value):
        pass

    def get_property(self, prop, obj, name):
        pass

    def load_object(self, obj):
        if not obj._loaded:
            obj = obj.get_by_id(obj.id)
            obj._loaded = True
        return obj