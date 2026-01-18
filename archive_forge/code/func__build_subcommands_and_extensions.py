import argparse
import csv
import glob
from importlib import util as importlib_util
import itertools
import logging
import os
import pkgutil
import sys
from oslo_utils import importutils
from manilaclient import api_versions
from manilaclient import client
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions as exc
import manilaclient.extension
from manilaclient.v2 import shell as shell_v2
def _build_subcommands_and_extensions(self, os_api_version, argv, options):
    self.extensions = self._discover_extensions(os_api_version)
    self._run_extension_hooks('__pre_parse_args__')
    self.parser = self.get_subcommand_parser(os_api_version.get_major_version())
    if argv and len(argv) > 1 and ('--help' in argv):
        argv = [x for x in argv if x != '--help']
        if argv[0] in self.subcommands:
            self.subcommands[argv[0]].print_help()
            return False
    if options.help or not argv:
        self.parser.print_help()
        return False
    args = self.parser.parse_args(argv)
    self._run_extension_hooks('__post_parse_args__', args)
    return args