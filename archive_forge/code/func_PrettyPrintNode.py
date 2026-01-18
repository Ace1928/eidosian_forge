import os
import sys
from xml.dom.minidom import parse
from xml.dom.minidom import Node
def PrettyPrintNode(node, indent=0):
    if node.nodeType == Node.TEXT_NODE:
        if node.data.strip():
            print('{}{}'.format(' ' * indent, node.data.strip()))
        return
    if node.childNodes:
        node.normalize()
    attr_count = 0
    if node.attributes:
        attr_count = node.attributes.length
    if attr_count == 0:
        print('{}<{}>'.format(' ' * indent, node.nodeName))
    else:
        print('{}<{}'.format(' ' * indent, node.nodeName))
        all_attributes = []
        for name, value in node.attributes.items():
            all_attributes.append((name, value))
            all_attributes.sort(CmpTuple())
        for name, value in all_attributes:
            print('{}  {}="{}"'.format(' ' * indent, name, value))
        print('%s>' % (' ' * indent))
    if node.nodeValue:
        print('{}  {}'.format(' ' * indent, node.nodeValue))
    for sub_node in node.childNodes:
        PrettyPrintNode(sub_node, indent=indent + 2)
    print('{}</{}>'.format(' ' * indent, node.nodeName))