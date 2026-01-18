import sys
from typing import Sequence
from twisted.application.app import _exitWithSignal
from twisted.internet.interfaces import IReactorCore, _ISupportsExitSignalCapturing
from twisted.python.usage import Options, UsageError
from ..runner._exit import ExitStatus, exit
from ..runner._runner import Runner
from ..service import Application, IService, IServiceMaker
from ._options import TwistOptions

        Executable entry point for L{Twist}.
        Processes options and run a twisted reactor with a service.

        @param argv: Command line arguments.
        @type argv: L{list}
        