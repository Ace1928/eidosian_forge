import os
from xdg.Menu import Menu, MenuEntry, Layout, Separator, XMLMenuBuilder
from xdg.BaseDirectory import xdg_config_dirs, xdg_data_dirs
from xdg.Exceptions import ParsingError 
from xdg.Config import setRootMode
def __deleteEntry(self, parent, entry, after=None, before=None):
    parent.Entries.remove(entry)
    xml_parent = self.__getXmlMenu(parent.getPath(True, True))
    if isinstance(entry, MenuEntry):
        entry.Parents.remove(parent)
        parent.MenuEntries.remove(entry)
        self.__addXmlFilename(xml_parent, entry.DesktopFileID, 'Exclude')
    elif isinstance(entry, Menu):
        parent.Submenus.remove(entry)
    if after or before:
        self.__addLayout(parent)
        self.__addXmlLayout(xml_parent, parent.Layout)