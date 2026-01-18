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
class _SubcommandParser(ParlaiParser):
    """
    ParlaiParser which always sets add_parlai_args and add_model_args to False.

    Used in the superscript to initialize just the args for that command.
    """

    def __init__(self, **kwargs):
        kwargs['add_parlai_args'] = False
        kwargs['add_model_args'] = False
        if 'description' not in kwargs:
            kwargs['description'] = None
        return super().__init__(**kwargs)

    def parse_known_args(self, args=None, namespace=None, nohelp=False):
        if not nohelp:
            self.add_extra_args(args)
        return super().parse_known_args(args, namespace, nohelp)