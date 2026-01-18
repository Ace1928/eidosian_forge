import os
from xdg.Menu import Menu, MenuEntry, Layout, Separator, XMLMenuBuilder
from xdg.BaseDirectory import xdg_config_dirs, xdg_data_dirs
from xdg.Exceptions import ParsingError 
from xdg.Config import setRootMode
def __addXmlTextElement(self, element, name, text):
    node = etree.SubElement(name, element)
    node.text = text
    return node