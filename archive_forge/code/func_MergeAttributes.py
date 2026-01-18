import os
import sys
from xml.dom.minidom import parse
from xml.dom.minidom import Node
def MergeAttributes(node1, node2):
    if not node2.attributes:
        return
    for name, value2 in node2.attributes.items():
        if name == 'Name':
            continue
        value1 = node1.getAttribute(name)
        if value1:
            if value1 != value2:
                node1.setAttribute(name, ';'.join([value1, value2]))
        else:
            node1.setAttribute(name, value2)
        if name == 'InheritedPropertySheets':
            node1.removeAttribute(name)