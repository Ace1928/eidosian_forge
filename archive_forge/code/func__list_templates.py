import os
import shutil
import string
from importlib import import_module
from pathlib import Path
from typing import Optional, cast
from urllib.parse import urlparse
import scrapy
from scrapy.commands import ScrapyCommand
from scrapy.exceptions import UsageError
from scrapy.utils.template import render_templatefile, string_camelcase
def _list_templates(self):
    print('Available templates:')
    for file in sorted(Path(self.templates_dir).iterdir()):
        if file.suffix == '.tmpl':
            print(f'  {file.stem}')