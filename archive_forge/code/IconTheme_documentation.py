import os, time
import re
from xdg.IniFile import IniFile, is_ascii
from xdg.BaseDirectory import xdg_data_dirs
from xdg.Exceptions import NoThemeError, debug
import xdg.Config
Retrieve the anchor points for overlays & emblems from the icon data,
        as a list of co-ordinate pairs, if they are specified.