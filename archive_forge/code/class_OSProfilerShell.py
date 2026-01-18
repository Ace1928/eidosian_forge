import argparse
import inspect
import sys
from oslo_config import cfg
import osprofiler
from osprofiler.cmd import commands
from osprofiler import exc
from osprofiler import opts
class OSProfilerShell(object):

    def __init__(self, argv):
        args = self._get_base_parser().parse_args(argv)
        opts.set_defaults(cfg.CONF)
        args.func(args)

    def _get_base_parser(self):
        parser = argparse.ArgumentParser(prog='osprofiler', description=__doc__.strip(), add_help=True)
        parser.add_argument('-v', '--version', action='version', version=osprofiler.__version__)
        self._append_subcommands(parser)
        return parser

    def _append_subcommands(self, parent_parser):
        subcommands = parent_parser.add_subparsers(help='<subcommands>')
        for group_cls in commands.BaseCommand.__subclasses__():
            group_parser = subcommands.add_parser(group_cls.group_name)
            subcommand_parser = group_parser.add_subparsers()
            for name, callback in inspect.getmembers(group_cls(), predicate=inspect.ismethod):
                command = name.replace('_', '-')
                desc = callback.__doc__ or ''
                help_message = desc.strip().split('\n')[0]
                arguments = getattr(callback, 'arguments', [])
                command_parser = subcommand_parser.add_parser(command, help=help_message, description=desc)
                for args, kwargs in arguments:
                    command_parser.add_argument(*args, **kwargs)
                command_parser.set_defaults(func=callback)

    def _no_project_and_domain_set(self, args):
        if not (args.os_project_id or (args.os_project_name and (args.os_user_domain_name or args.os_user_domain_id)) or (args.os_tenant_id or args.os_tenant_name)):
            return True
        else:
            return False