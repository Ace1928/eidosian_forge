import getopt
import inspect
import itertools
from types import MethodType
from typing import Dict, List, Set
from twisted.python import reflect, usage, util
from twisted.python.compat import ioType
class SubcommandAction(usage.Completer):

    def _shellCode(self, optName, shellType):
        if shellType == usage._ZSH:
            return '*::subcmd:->subcmd'
        raise NotImplementedError(f'Unknown shellType {shellType!r}')