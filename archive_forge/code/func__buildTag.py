from xml.sax.handler import ContentHandler
from lxml import etree
from lxml.etree import ElementTree, SubElement
from lxml.etree import Comment, ProcessingInstruction
def _buildTag(self, ns_name_tuple):
    ns_uri, local_name = ns_name_tuple
    if ns_uri:
        el_tag = '{%s}%s' % ns_name_tuple
    elif self._default_ns:
        el_tag = '{%s}%s' % (self._default_ns, local_name)
    else:
        el_tag = local_name
    return el_tag