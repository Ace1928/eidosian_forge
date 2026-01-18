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
def _module_exists(module_name):
    spec = find_spec(module_name)
    return spec is not None and spec.loader is not None