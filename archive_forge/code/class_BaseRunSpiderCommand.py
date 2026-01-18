import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from twisted.python import failure
from scrapy.crawler import CrawlerProcess
from scrapy.exceptions import UsageError
from scrapy.utils.conf import arglist_to_dict, feed_process_params_from_cli
class BaseRunSpiderCommand(ScrapyCommand):
    """
    Common class used to share functionality between the crawl, parse and runspider commands
    """

    def add_options(self, parser):
        ScrapyCommand.add_options(self, parser)
        parser.add_argument('-a', dest='spargs', action='append', default=[], metavar='NAME=VALUE', help='set spider argument (may be repeated)')
        parser.add_argument('-o', '--output', metavar='FILE', action='append', help='append scraped items to the end of FILE (use - for stdout), to define format set a colon at the end of the output URI (i.e. -o FILE:FORMAT)')
        parser.add_argument('-O', '--overwrite-output', metavar='FILE', action='append', help='dump scraped items into FILE, overwriting any existing file, to define format set a colon at the end of the output URI (i.e. -O FILE:FORMAT)')
        parser.add_argument('-t', '--output-format', metavar='FORMAT', help='format to use for dumping items')

    def process_options(self, args, opts):
        ScrapyCommand.process_options(self, args, opts)
        try:
            opts.spargs = arglist_to_dict(opts.spargs)
        except ValueError:
            raise UsageError('Invalid -a value, use -a NAME=VALUE', print_help=False)
        if opts.output or opts.overwrite_output:
            feeds = feed_process_params_from_cli(self.settings, opts.output, opts.output_format, opts.overwrite_output)
            self.settings.set('FEEDS', feeds, priority='cmdline')