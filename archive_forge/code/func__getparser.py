import argparse
from gettext import gettext
import os
import sys
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import final
from typing import List
from typing import Literal
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
import _pytest._io
from _pytest.config.exceptions import UsageError
from _pytest.deprecated import check_ispytest
def _getparser(self) -> 'MyOptionParser':
    from _pytest._argcomplete import filescompleter
    optparser = MyOptionParser(self, self.extra_info, prog=self.prog)
    groups = [*self._groups, self._anonymous]
    for group in groups:
        if group.options:
            desc = group.description or group.name
            arggroup = optparser.add_argument_group(desc)
            for option in group.options:
                n = option.names()
                a = option.attrs()
                arggroup.add_argument(*n, **a)
    file_or_dir_arg = optparser.add_argument(FILE_OR_DIR, nargs='*')
    file_or_dir_arg.completer = filescompleter
    return optparser