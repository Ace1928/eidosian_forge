import io
import argparse
from typing import List, Optional, Dict, Any
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser, CustomHelpFormatter
from abc import abstractmethod
import importlib
import pkgutil
import parlai.scripts
import parlai.utils.logging as logging
from parlai.core.loader import register_script, SCRIPT_REGISTRY  # noqa: F401
class _SupercommandParser(ParlaiParser):
    """
    Specialty ParlAI parser used for the supercommand.

    Contains some special behavior.
    """

    def __init__(self, *args, **kwargs):
        from parlai.utils.strings import colorize
        logo = ''
        logo += colorize('       _', 'red') + '\n'
        logo += colorize('      /', 'red') + colorize('"', 'brightblack')
        logo += colorize(')', 'yellow') + '\n'
        logo += colorize('     //', 'red') + colorize(')', 'yellow') + '\n'
        logo += colorize('  ==', 'green')
        logo += colorize('/', 'blue') + colorize('/', 'red') + colorize("'", 'yellow')
        logo += colorize('===', 'green') + ' ParlAI\n'
        logo += colorize('   /', 'blue')
        kwargs['description'] = logo
        return super().__init__(*args, **kwargs)

    def add_extra_args(self, args):
        sa = [a for a in self._actions if isinstance(a, argparse._SubParsersAction)]
        assert len(sa) == 1
        sa = sa[0]
        for _, v in sa.choices.items():
            v.add_extra_args(args)

    def add_subparsers(self, **kwargs):
        return super().add_subparsers(**kwargs)

    def _unsuppress_hidden(self):
        """
        Restore the help messages of hidden commands.
        """
        spa = [a for a in self._actions if isinstance(a, argparse._SubParsersAction)]
        assert len(spa) == 1
        spa = spa[0]
        for choices_action in spa._choices_actions:
            dest = choices_action.dest
            if choices_action.help == argparse.SUPPRESS:
                choices_action.help = spa.choices[dest].description

    def print_helpall(self):
        self._unsuppress_hidden()
        self.print_help()