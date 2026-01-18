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
def _get_commands_from_module(module, inproject):
    d = {}
    for cmd in _iter_command_classes(module):
        if inproject or not cmd.requires_project:
            cmdname = cmd.__module__.split('.')[-1]
            d[cmdname] = cmd()
    return d