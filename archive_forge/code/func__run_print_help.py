import argparse
import cProfile
import inspect
import os
import sys
from importlib.metadata import entry_points
import scrapy
from scrapy.commands import BaseRunSpiderCommand, ScrapyCommand, ScrapyHelpFormatter
from scrapy.crawler import CrawlerProcess
from scrapy.exceptions import UsageError
from scrapy.utils.misc import walk_modules
from scrapy.utils.project import get_project_settings, inside_project
from scrapy.utils.python import garbage_collect
def _run_print_help(parser, func, *a, **kw):
    try:
        func(*a, **kw)
    except UsageError as e:
        if str(e):
            parser.error(str(e))
        if e.print_help:
            parser.print_help()
        sys.exit(2)