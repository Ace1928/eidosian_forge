import pkg_resources
import sys
import optparse
from . import bool_optparse
import os
import re
import textwrap
from . import pluginlib
import configparser
import getpass
from logging.config import fileConfig
class NotFoundCommand(Command):

    def run(self, args):
        print('Command %r not known (you may need to run setup.py egg_info)' % self.command_name)
        commands = sorted(get_commands().items())
        if not commands:
            print('No commands registered.')
            print('Have you installed Paste Script?')
            print('(try running python setup.py develop)')
            return 2
        print('Known commands:')
        longest = max([len(n) for n, c in commands])
        for name, command in commands:
            print('  %s  %s' % (self.pad(name, length=longest), command.load().summary))
        return 2