import os
import re
import stat
import sys
import fnmatch
from xdg import BaseDirectory
import xdg.Locale
from xml.dom import minidom, XML_NAMESPACE
from collections import defaultdict
class DiscardMagicRules(Exception):
    """Raised when __NOMAGIC__ is found, and caught to discard previous rules."""
    pass