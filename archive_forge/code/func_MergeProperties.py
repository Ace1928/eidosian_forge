import os
import sys
from xml.dom.minidom import parse
from xml.dom.minidom import Node
def MergeProperties(node1, node2):
    MergeAttributes(node1, node2)
    for child2 in node2.childNodes:
        child1 = SeekToNode(node1, child2)
        if child1:
            MergeProperties(child1, child2)
        else:
            node1.appendChild(child2.cloneNode(True))