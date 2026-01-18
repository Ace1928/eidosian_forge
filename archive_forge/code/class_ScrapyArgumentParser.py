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
class ScrapyArgumentParser(argparse.ArgumentParser):

    def _parse_optional(self, arg_string):
        if arg_string[:2] == '-:':
            return None
        return super()._parse_optional(arg_string)