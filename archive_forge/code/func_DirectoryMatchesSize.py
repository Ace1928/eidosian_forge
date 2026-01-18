import os, time
import re
from xdg.IniFile import IniFile, is_ascii
from xdg.BaseDirectory import xdg_data_dirs
from xdg.Exceptions import NoThemeError, debug
import xdg.Config
def DirectoryMatchesSize(subdir, iconsize, theme):
    Type = theme.getType(subdir)
    Size = theme.getSize(subdir)
    Threshold = theme.getThreshold(subdir)
    MinSize = theme.getMinSize(subdir)
    MaxSize = theme.getMaxSize(subdir)
    if Type == 'Fixed':
        return Size == iconsize
    elif Type == 'Scaleable':
        return MinSize <= iconsize <= MaxSize
    elif Type == 'Threshold':
        return Size - Threshold <= iconsize <= Size + Threshold