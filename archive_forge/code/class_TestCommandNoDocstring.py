import argparse
import functools
from cliff import command
from cliff.tests import base
class TestCommandNoDocstring(command.Command):

    def take_action(self, parsed_args):
        return 42