from xml.parsers import expat
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesImpl
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