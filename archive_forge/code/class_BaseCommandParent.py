import pkg_resources
import argparse
import logging
import sys
from warnings import warn
class BaseCommandParent(object):
    """
    A base interface for Pecan commands.

    Can be extended to support ``pecan`` command extensions in individual Pecan
    projects, e.g.,

    $ ``pecan my-custom-command config.py``

    ::

        # myapp/myapp/custom_command.py
        class CustomCommand(pecan.commands.base.BaseCommand):
            '''
            (First) line of the docstring is used to summarize the command.
            '''

            arguments = ({
                'name': '--extra_arg',
                'help': 'an extra command line argument',
                'optional': True
            })

            def run(self, args):
                super(SomeCommand, self).run(args)
                if args.extra_arg:
                    pass
    """
    arguments = ({'name': 'config_file', 'help': 'a Pecan configuration file', 'nargs': '?', 'default': None},)

    def run(self, args):
        """To be implemented by subclasses."""
        self.args = args

    def load_app(self):
        from pecan import load_app
        return load_app(self.args.config_file)