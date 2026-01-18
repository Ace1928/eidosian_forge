import ast
import logging
from cliff import command
from cliff.formatters import table
from cliff import lister
from cliff import show
from blazarclient import exception
from blazarclient import utils
class OpenStackCommand(command.Command):
    """Base class for OpenStack commands."""
    api = None

    def run(self, parsed_args):
        if not self.api:
            return
        else:
            return super(OpenStackCommand, self).run(parsed_args)

    def get_data(self, parsed_args):
        pass

    def take_action(self, parsed_args):
        return self.get_data(parsed_args)