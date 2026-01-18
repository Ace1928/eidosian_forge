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
def _find_template(self, template: str) -> Optional[Path]:
    template_file = Path(self.templates_dir, f'{template}.tmpl')
    if template_file.exists():
        return template_file
    print(f'Unable to find template: {template}\n')
    print('Use "scrapy genspider --list" to see all available templates.')
    return None