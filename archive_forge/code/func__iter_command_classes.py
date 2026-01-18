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
def _iter_command_classes(module_name):
    for module in walk_modules(module_name):
        for obj in vars(module).values():
            if inspect.isclass(obj) and issubclass(obj, ScrapyCommand) and (obj.__module__ == module.__name__) and (obj not in (ScrapyCommand, BaseRunSpiderCommand)):
                yield obj