import os
import re
import string
from importlib.util import find_spec
from pathlib import Path
from shutil import copy2, copystat, ignore_patterns, move
from stat import S_IWUSR as OWNER_WRITE_PERMISSION
import scrapy
from scrapy.commands import ScrapyCommand
from scrapy.exceptions import UsageError
from scrapy.utils.template import render_templatefile, string_camelcase
def _make_writable(path):
    current_permissions = os.stat(path).st_mode
    os.chmod(path, current_permissions | OWNER_WRITE_PERMISSION)