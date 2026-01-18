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
def _print_commands(settings, inproject):
    _print_header(settings, inproject)
    print('Usage:')
    print('  scrapy <command> [options] [args]\n')
    print('Available commands:')
    cmds = _get_commands_dict(settings, inproject)
    for cmdname, cmdclass in sorted(cmds.items()):
        print(f'  {cmdname:<13} {cmdclass.short_desc()}')
    if not inproject:
        print()
        print('  [ more ]      More commands available when run from project directory')
    print()
    print('Use "scrapy <command> -h" to see more info about a command')