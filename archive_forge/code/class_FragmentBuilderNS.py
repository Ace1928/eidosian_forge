from a string or file.
from xml.dom import xmlbuilder, minidom, Node
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE
from xml.parsers import expat
from xml.dom.minidom import _append_child, _set_attribute_node
from xml.dom.NodeFilter import NodeFilter
class FragmentBuilderNS(Namespaces, FragmentBuilder):
    """Fragment builder that supports namespaces."""

    def reset(self):
        FragmentBuilder.reset(self)
        self._initNamespaces()

    def _getNSattrs(self):
        """Return string of namespace attributes from this element and
        ancestors."""
        attrs = ''
        context = self.context
        L = []
        while context:
            if hasattr(context, '_ns_prefix_uri'):
                for prefix, uri in context._ns_prefix_uri.items():
                    if prefix in L:
                        continue
                    L.append(prefix)
                    if prefix:
                        declname = 'xmlns:' + prefix
                    else:
                        declname = 'xmlns'
                    if attrs:
                        attrs = "%s\n    %s='%s'" % (attrs, declname, uri)
                    else:
                        attrs = " %s='%s'" % (declname, uri)
            context = context.parentNode
        return attrs