from __future__ import annotations
from collections import defaultdict
import datetime
import json
from typing import (
from pandas.io.excel._base import ExcelWriter
from pandas.io.excel._util import (
def _create_freeze_panes(self, sheet_name: str, freeze_panes: tuple[int, int]) -> None:
    """
        Create freeze panes in the sheet.

        Parameters
        ----------
        sheet_name : str
            Name of the spreadsheet
        freeze_panes : tuple of (int, int)
            Freeze pane location x and y
        """
    from odf.config import ConfigItem, ConfigItemMapEntry, ConfigItemMapIndexed, ConfigItemMapNamed, ConfigItemSet
    config_item_set = ConfigItemSet(name='ooo:view-settings')
    self.book.settings.addElement(config_item_set)
    config_item_map_indexed = ConfigItemMapIndexed(name='Views')
    config_item_set.addElement(config_item_map_indexed)
    config_item_map_entry = ConfigItemMapEntry()
    config_item_map_indexed.addElement(config_item_map_entry)
    config_item_map_named = ConfigItemMapNamed(name='Tables')
    config_item_map_entry.addElement(config_item_map_named)
    config_item_map_entry = ConfigItemMapEntry(name=sheet_name)
    config_item_map_named.addElement(config_item_map_entry)
    config_item_map_entry.addElement(ConfigItem(name='HorizontalSplitMode', type='short', text='2'))
    config_item_map_entry.addElement(ConfigItem(name='VerticalSplitMode', type='short', text='2'))
    config_item_map_entry.addElement(ConfigItem(name='HorizontalSplitPosition', type='int', text=str(freeze_panes[0])))
    config_item_map_entry.addElement(ConfigItem(name='VerticalSplitPosition', type='int', text=str(freeze_panes[1])))
    config_item_map_entry.addElement(ConfigItem(name='PositionRight', type='int', text=str(freeze_panes[0])))
    config_item_map_entry.addElement(ConfigItem(name='PositionBottom', type='int', text=str(freeze_panes[1])))