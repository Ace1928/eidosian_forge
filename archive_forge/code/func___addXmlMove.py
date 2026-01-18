import os
from xdg.Menu import Menu, MenuEntry, Layout, Separator, XMLMenuBuilder
from xdg.BaseDirectory import xdg_config_dirs, xdg_data_dirs
from xdg.Exceptions import ParsingError 
from xdg.Config import setRootMode
def __addXmlMove(self, element, old, new):
    node = etree.SubElement('Move', element)
    self.__addXmlTextElement(node, 'Old', old)
    self.__addXmlTextElement(node, 'New', new)
    return node