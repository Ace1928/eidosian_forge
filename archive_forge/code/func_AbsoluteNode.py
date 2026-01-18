import os
import sys
from xml.dom.minidom import parse
from xml.dom.minidom import Node
def AbsoluteNode(node):
    """Makes all the properties we know about in this node absolute."""
    if node.attributes:
        for name, value in node.attributes.items():
            if name in ['InheritedPropertySheets', 'RelativePath', 'AdditionalIncludeDirectories', 'IntermediateDirectory', 'OutputDirectory', 'AdditionalLibraryDirectories']:
                path_list = value.split(';')
                new_list = FixFilenames(path_list, os.path.dirname(ARGUMENTS[1]))
                node.setAttribute(name, ';'.join(new_list))
            if not value:
                node.removeAttribute(name)