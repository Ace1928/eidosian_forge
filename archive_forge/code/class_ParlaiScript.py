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
class ParlaiScript(object):
    """
    A ParlAI script is a standardized form of access.
    """
    parser: ParlaiParser

    @classmethod
    @abstractmethod
    def setup_args(cls) -> ParlaiParser:
        """
        Create the parser with args.
        """
        pass

    def __init__(self, opt: Opt):
        self.opt = opt

    @abstractmethod
    def run(self):
        """
        The main method.

        Must be implemented by the script writer.
        """
        raise NotImplementedError()

    @classmethod
    def _run_kwargs(cls, kwargs: Dict[str, Any]):
        """
        Construct and run the script using kwargs, pseudo-parsing them.
        """
        parser = cls.setup_args()
        opt = parser.parse_kwargs(**kwargs)
        return cls._run_from_parser_and_opt(opt, parser)

    @classmethod
    def _run_args(cls, args: Optional[List[str]]=None):
        """
        Construct and run the script using args, defaulting to getting from CLI.
        """
        parser = cls.setup_args()
        opt = parser.parse_args(args=args)
        return cls._run_from_parser_and_opt(opt, parser)

    @classmethod
    def _run_from_parser_and_opt(cls, opt: Opt, parser: ParlaiParser):
        script = cls(opt)
        script.parser = parser
        return script.run()

    @classmethod
    def main(cls, *args, **kwargs):
        """
        Run the program, possibly with some given args.

        You may provide command line args in the form of strings, or
        options. For example:

        >>> MyScript.main(['--task', 'convai2'])
        >>> MyScript.main(task='convai2')

        You may not combine both args and kwargs.
        """
        assert not (bool(args) and bool(kwargs))
        if args:
            return cls._run_args(args)
        elif kwargs:
            return cls._run_kwargs(kwargs)
        else:
            return cls._run_args(None)

    @classmethod
    def help(cls, **kwargs):
        f = io.StringIO()
        parser = cls.setup_args()
        parser.prog = cls.__name__
        parser.add_extra_args(parser._kwargs_to_str_args(**kwargs))
        parser.print_help(f)
        return f.getvalue()