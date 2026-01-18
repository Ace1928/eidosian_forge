import os
import sys
from xml.dom.minidom import parse
from xml.dom.minidom import Node
def GetConfiguationNodes(vcproj):
    nodes = []
    for node in vcproj.childNodes:
        if node.nodeName == 'Configurations':
            for sub_node in node.childNodes:
                if sub_node.nodeName == 'Configuration':
                    nodes.append(sub_node)
    return nodes