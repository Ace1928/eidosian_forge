import os
from xdg.Menu import Menu, MenuEntry, Layout, Separator, XMLMenuBuilder
from xdg.BaseDirectory import xdg_config_dirs, xdg_data_dirs
from xdg.Exceptions import ParsingError 
from xdg.Config import setRootMode
def __saveEntries(self, menu):
    if not menu:
        menu = self.menu
    if isinstance(menu.Directory, MenuEntry):
        menu.Directory.save()
    for entry in menu.getEntries(hidden=True):
        if isinstance(entry, MenuEntry):
            entry.save()
        elif isinstance(entry, Menu):
            self.__saveEntries(entry)