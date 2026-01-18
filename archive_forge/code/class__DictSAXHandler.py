from xml.parsers import expat
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesImpl
class _DictSAXHandler(object):

    def __init__(self, item_depth=0, item_callback=lambda *args: True, xml_attribs=True, attr_prefix='@', cdata_key='#text', force_cdata=False, cdata_separator='', postprocessor=None, dict_constructor=OrderedDict, strip_whitespace=True, namespace_separator=':', namespaces=None, force_list=()):
        self.path = []
        self.stack = []
        self.data = []
        self.item = None
        self.item_depth = item_depth
        self.xml_attribs = xml_attribs
        self.item_callback = item_callback
        self.attr_prefix = attr_prefix
        self.cdata_key = cdata_key
        self.force_cdata = force_cdata
        self.cdata_separator = cdata_separator
        self.postprocessor = postprocessor
        self.dict_constructor = dict_constructor
        self.strip_whitespace = strip_whitespace
        self.namespace_separator = namespace_separator
        self.namespaces = namespaces
        self.force_list = force_list

    def _build_name(self, full_name):
        if not self.namespaces:
            return full_name
        i = full_name.rfind(self.namespace_separator)
        if i == -1:
            return full_name
        namespace, name = (full_name[:i], full_name[i + 1:])
        short_namespace = self.namespaces.get(namespace, namespace)
        if not short_namespace:
            return name
        else:
            return self.namespace_separator.join((short_namespace, name))

    def _attrs_to_dict(self, attrs):
        if isinstance(attrs, dict):
            return attrs
        return self.dict_constructor(zip(attrs[0::2], attrs[1::2]))

    def startElement(self, full_name, attrs):
        name = self._build_name(full_name)
        attrs = self._attrs_to_dict(attrs)
        self.path.append((name, attrs or None))
        if len(self.path) > self.item_depth:
            self.stack.append((self.item, self.data))
            if self.xml_attribs:
                attrs = self.dict_constructor(((self.attr_prefix + self._build_name(key), value) for key, value in attrs.items()))
            else:
                attrs = None
            self.item = attrs or None
            self.data = []

    def endElement(self, full_name):
        name = self._build_name(full_name)
        if len(self.path) == self.item_depth:
            item = self.item
            if item is None:
                item = None if not self.data else self.cdata_separator.join(self.data)
            should_continue = self.item_callback(self.path, item)
            if not should_continue:
                raise ParsingInterrupted()
        if len(self.stack):
            data = None if not self.data else self.cdata_separator.join(self.data)
            item = self.item
            self.item, self.data = self.stack.pop()
            if self.strip_whitespace and data:
                data = data.strip() or None
            if data and self.force_cdata and (item is None):
                item = self.dict_constructor()
            if item is not None:
                if data:
                    self.push_data(item, self.cdata_key, data)
                self.item = self.push_data(self.item, name, item)
            else:
                self.item = self.push_data(self.item, name, data)
        else:
            self.item = None
            self.data = []
        self.path.pop()

    def characters(self, data):
        if not self.data:
            self.data = [data]
        else:
            self.data.append(data)

    def push_data(self, item, key, data):
        if self.postprocessor is not None:
            result = self.postprocessor(self.path, key, data)
            if result is None:
                return item
            key, data = result
        if item is None:
            item = self.dict_constructor()
        try:
            value = item[key]
            if isinstance(value, list):
                value.append(data)
            else:
                item[key] = [value, data]
        except KeyError:
            if key in self.force_list:
                item[key] = [data]
            else:
                item[key] = data
        return item