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
def _genspider(self, module, name, url, template_name, template_file):
    """Generate the spider module, based on the given template"""
    capitalized_module = ''.join((s.capitalize() for s in module.split('_')))
    domain = extract_domain(url)
    tvars = {'project_name': self.settings.get('BOT_NAME'), 'ProjectName': string_camelcase(self.settings.get('BOT_NAME')), 'module': module, 'name': name, 'url': url, 'domain': domain, 'classname': f'{capitalized_module}Spider'}
    if self.settings.get('NEWSPIDER_MODULE'):
        spiders_module = import_module(self.settings['NEWSPIDER_MODULE'])
        spiders_dir = Path(spiders_module.__file__).parent.resolve()
    else:
        spiders_module = None
        spiders_dir = Path('.')
    spider_file = f'{spiders_dir / module}.py'
    shutil.copyfile(template_file, spider_file)
    render_templatefile(spider_file, **tvars)
    print(f'Created spider {name!r} using template {template_name!r} ', end='' if spiders_module else '\n')
    if spiders_module:
        print(f'in module:\n  {spiders_module.__name__}.{module}')