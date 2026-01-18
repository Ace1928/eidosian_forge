import argparse
import importlib
import logging
import os
import subprocess
import sys
from pathlib import Path
import psutil
import torch
from accelerate.commands.config import default_config_file, load_config_from_file
from accelerate.commands.config.config_args import SageMakerConfig
from accelerate.commands.config.config_utils import DYNAMO_BACKENDS
from accelerate.commands.utils import CustomArgumentParser
from accelerate.state import get_int_from_env
from accelerate.utils import (
from accelerate.utils.constants import DEEPSPEED_MULTINODE_LAUNCHERS, TORCH_DYNAMO_MODES
class CustomHelpFormatter(argparse.HelpFormatter):
    """
    This is a custom help formatter that will hide all arguments that are not used in the command line when the help is
    called. This is useful for the case where the user is using a specific platform and only wants to see the arguments
    for that platform.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.titles = ['Hardware Selection Arguments', 'Resource Selection Arguments', 'Training Paradigm Arguments', 'positional arguments', 'optional arguments']

    def add_argument(self, action: argparse.Action):
        if 'accelerate' in sys.argv[0] and 'launch' in sys.argv[1:]:
            args = sys.argv[2:]
        else:
            args = sys.argv[1:]
        if len(args) > 1:
            args = list(map(clean_option, args))
            used_platforms = [arg for arg in args if arg in options_to_group.keys()]
            used_titles = [options_to_group[o] for o in used_platforms]
            if action.container.title not in self.titles + used_titles:
                action.help = argparse.SUPPRESS
            elif action.container.title == 'Hardware Selection Arguments':
                if set(action.option_strings).isdisjoint(set(args)):
                    action.help = argparse.SUPPRESS
                else:
                    action.help = action.help + ' (currently selected)'
            elif action.container.title == 'Training Paradigm Arguments':
                if set(action.option_strings).isdisjoint(set(args)):
                    action.help = argparse.SUPPRESS
                else:
                    action.help = action.help + ' (currently selected)'
        action.option_strings = [s for s in action.option_strings if '-' not in s[2:]]
        super().add_argument(action)

    def end_section(self):
        if len(self._current_section.items) < 2:
            self._current_section.items = []
            self._current_section.heading = ''
        super().end_section()