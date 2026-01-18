import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
class cmd_Demo(commands.Command):
    __doc__ = 'A sample command.\n\n            :Usage:\n                bzr demo\n\n            :Examples:\n                Example 1::\n\n                    cmd arg1\n\n            Blah Blah Blah\n            '