from argparse import _SubParsersAction
from ..utils import dump_environment_info
from . import BaseHuggingfaceCLICommand
class EnvironmentCommand(BaseHuggingfaceCLICommand):

    def __init__(self, args):
        self.args = args

    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        env_parser = parser.add_parser('env', help='Print information about the environment.')
        env_parser.set_defaults(func=EnvironmentCommand)

    def run(self) -> None:
        dump_environment_info()