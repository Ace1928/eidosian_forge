from configparser import ConfigParser
import os
import re
import sys
from urllib.parse import unquote
from paste.deploy.util import fix_call, importlib_metadata, lookup_object
def _loadegg(object_type, uri, spec, name, relative_to, global_conf):
    loader = EggLoader(spec)
    return loader.get_context(object_type, name, global_conf)