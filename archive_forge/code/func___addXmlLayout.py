import os
from xdg.Menu import Menu, MenuEntry, Layout, Separator, XMLMenuBuilder
from xdg.BaseDirectory import xdg_config_dirs, xdg_data_dirs
from xdg.Exceptions import ParsingError 
from xdg.Config import setRootMode
def __addXmlLayout(self, element, layout):
    for node in element.findall('Layout'):
        element.remove(node)
    node = etree.SubElement('Layout', element)
    for order in layout.order:
        if order[0] == 'Separator':
            child = etree.SubElement('Separator', node)
        elif order[0] == 'Filename':
            child = self.__addXmlTextElement(node, 'Filename', order[1])
        elif order[0] == 'Menuname':
            child = self.__addXmlTextElement(node, 'Menuname', order[1])
        elif order[0] == 'Merge':
            child = etree.SubElement('Merge', node)
            child.attrib['type'] = order[1]
    return node