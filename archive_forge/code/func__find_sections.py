from configparser import ConfigParser
import os
import re
import sys
from urllib.parse import unquote
from paste.deploy.util import fix_call, importlib_metadata, lookup_object
def _find_sections(self, sections, name_prefix, name):
    found = []
    if name is None:
        if name_prefix in sections:
            found.append(name_prefix)
        name = 'main'
    for section in sections:
        if section.startswith(name_prefix + ':'):
            if section[len(name_prefix) + 1:].strip() == name:
                found.append(section)
    return found