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
class ManilaClientArgumentParser(argparse.ArgumentParser):

    def __init__(self, *args, **kwargs):
        super(ManilaClientArgumentParser, self).__init__(*args, **kwargs)
        self.register('action', 'single_alias', AllowOnlyOneAliasAtATimeAction)

    def error(self, message):
        """error(message: string)

        Prints a usage message incorporating the message to stderr and
        exits.
        """
        self.print_usage(sys.stderr)
        choose_from = ' (choose from'
        progparts = self.prog.partition(' ')
        self.exit(2, "error: %(errmsg)s\nTry '%(mainp)s help %(subp)s' for more information.\n" % {'errmsg': message.split(choose_from)[0], 'mainp': progparts[0], 'subp': progparts[2]})

    def _get_option_tuples(self, option_string):
        """Avoid ambiguity in argument abbreviation.

        Manilaclient uses aliases for command parameters and this method
        is used for avoiding parameter ambiguity alert.
        """
        option_tuples = super(ManilaClientArgumentParser, self)._get_option_tuples(option_string)
        if len(option_tuples) > 1:
            opt_strings_list = []
            opts = []
            for opt in option_tuples:
                if opt[0].option_strings not in opt_strings_list:
                    opt_strings_list.append(opt[0].option_strings)
                    opts.append(opt)
            return opts
        return option_tuples