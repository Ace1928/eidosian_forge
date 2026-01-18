import abc
import argparse
import os
from zunclient.common.apiclient import exceptions
@staticmethod
def _parser_add_opt(parser, opt):
    """Add an option to parser in two variants.

        :param opt: option name (with underscores)
        """
    dashed_opt = opt.replace('_', '-')
    env_var = 'OS_%s' % opt.upper()
    arg_default = os.environ.get(env_var, '')
    arg_help = 'Defaults to env[%s].' % env_var
    parser.add_argument('--os-%s' % dashed_opt, metavar='<%s>' % dashed_opt, default=arg_default, help=arg_help)
    parser.add_argument('--os_%s' % opt, metavar='<%s>' % dashed_opt, help=argparse.SUPPRESS)