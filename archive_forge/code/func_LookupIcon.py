import os, time
import re
from xdg.IniFile import IniFile, is_ascii
from xdg.BaseDirectory import xdg_data_dirs
from xdg.Exceptions import NoThemeError, debug
import xdg.Config
def LookupIcon(iconname, size, theme, extensions):
    if theme.name not in theme_cache:
        theme_cache[theme.name] = []
        theme_cache[theme.name].append(time.time() - (xdg.Config.cache_time + 1))
        theme_cache[theme.name].append(0)
        theme_cache[theme.name].append(dict())
    if int(time.time() - theme_cache[theme.name][0]) >= xdg.Config.cache_time:
        theme_cache[theme.name][0] = time.time()
        for subdir in theme.getDirectories():
            for directory in icondirs:
                dir = os.path.join(directory, theme.name, subdir)
                if (dir not in theme_cache[theme.name][2] or theme_cache[theme.name][1] < os.path.getmtime(os.path.join(directory, theme.name))) and subdir != '' and os.path.isdir(dir):
                    theme_cache[theme.name][2][dir] = [subdir, os.listdir(dir)]
                    theme_cache[theme.name][1] = os.path.getmtime(os.path.join(directory, theme.name))
    for dir, values in theme_cache[theme.name][2].items():
        if DirectoryMatchesSize(values[0], size, theme):
            for extension in extensions:
                if iconname + '.' + extension in values[1]:
                    return os.path.join(dir, iconname + '.' + extension)
    minimal_size = 2 ** 31
    closest_filename = ''
    for dir, values in theme_cache[theme.name][2].items():
        distance = DirectorySizeDistance(values[0], size, theme)
        if distance < minimal_size:
            for extension in extensions:
                if iconname + '.' + extension in values[1]:
                    closest_filename = os.path.join(dir, iconname + '.' + extension)
                    minimal_size = distance
    return closest_filename