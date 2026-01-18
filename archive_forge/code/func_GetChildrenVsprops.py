import os
import sys
from xml.dom.minidom import parse
from xml.dom.minidom import Node
def GetChildrenVsprops(filename):
    dom = parse(filename)
    if dom.documentElement.attributes:
        vsprops = dom.documentElement.getAttribute('InheritedPropertySheets')
        return FixFilenames(vsprops.split(';'), os.path.dirname(filename))
    return []