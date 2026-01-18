import string
from xml.dom import Node
def _inclusiveNamespacePrefixes(node, context, unsuppressedPrefixes):
    """http://www.w3.org/TR/xml-exc-c14n/ 
    InclusiveNamespaces PrefixList parameter, which lists namespace prefixes that 
    are handled in the manner described by the Canonical XML Recommendation"""
    inclusive = []
    if node.prefix:
        usedPrefixes = ['xmlns:%s' % node.prefix]
    else:
        usedPrefixes = ['xmlns']
    for a in _attrs(node):
        if a.nodeName.startswith('xmlns') or not a.prefix:
            continue
        usedPrefixes.append('xmlns:%s' % a.prefix)
    unused_namespace_dict = {}
    for attr in context:
        n = attr.nodeName
        if n in unsuppressedPrefixes:
            inclusive.append(attr)
        elif n.startswith('xmlns:') and n[6:] in unsuppressedPrefixes:
            inclusive.append(attr)
        elif n.startswith('xmlns') and n[5:] in unsuppressedPrefixes:
            inclusive.append(attr)
        elif attr.nodeName in usedPrefixes:
            inclusive.append(attr)
        elif n.startswith('xmlns:'):
            unused_namespace_dict[n] = attr.value
    return (inclusive, unused_namespace_dict)