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
def _addoption_instance(self, option: 'Argument', shortupper: bool=False) -> None:
    if not shortupper:
        for opt in option._short_opts:
            if opt[0] == '-' and opt[1].islower():
                raise ValueError('lowercase shortoptions reserved')
    if self.parser:
        self.parser.processoption(option)
    self.options.append(option)