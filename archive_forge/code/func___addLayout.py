import os
from xdg.Menu import Menu, MenuEntry, Layout, Separator, XMLMenuBuilder
from xdg.BaseDirectory import xdg_config_dirs, xdg_data_dirs
from xdg.Exceptions import ParsingError 
from xdg.Config import setRootMode
def __addLayout(self, parent):
    layout = Layout()
    layout.order = []
    layout.show_empty = parent.Layout.show_empty
    layout.inline = parent.Layout.inline
    layout.inline_header = parent.Layout.inline_header
    layout.inline_alias = parent.Layout.inline_alias
    layout.inline_limit = parent.Layout.inline_limit
    layout.order.append(['Merge', 'menus'])
    for entry in parent.Entries:
        if isinstance(entry, Menu):
            layout.parseMenuname(entry.Name)
        elif isinstance(entry, MenuEntry):
            layout.parseFilename(entry.DesktopFileID)
        elif isinstance(entry, Separator):
            layout.parseSeparator()
    layout.order.append(['Merge', 'files'])
    parent.Layout = layout
    return layout