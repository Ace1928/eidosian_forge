import os
import re
import stat
import sys
import fnmatch
from xdg import BaseDirectory
import xdg.Locale
from xml.dom import minidom, XML_NAMESPACE
from collections import defaultdict
def _cache_database():
    global globs, magic, aliases, inheritance, _cache_uptodate
    _cache_uptodate = True
    aliases = {}
    inheritance = defaultdict(set)
    for path in BaseDirectory.load_data_paths(os.path.join('mime', 'aliases')):
        with open(path, 'r') as f:
            for line in f:
                alias, canonical = line.strip().split(None, 1)
                aliases[alias] = canonical
    globs = GlobDB()
    for path in BaseDirectory.load_data_paths(os.path.join('mime', 'globs2')):
        globs.merge_file(path)
    globs.finalise()
    magic = MagicDB()
    for path in BaseDirectory.load_data_paths(os.path.join('mime', 'magic')):
        magic.merge_file(path)
    magic.finalise()
    for path in BaseDirectory.load_data_paths(os.path.join('mime', 'subclasses')):
        with open(path, 'r') as f:
            for line in f:
                sub, parent = line.strip().split(None, 1)
                inheritance[sub].add(parent)