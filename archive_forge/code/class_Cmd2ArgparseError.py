from typing import (
class Cmd2ArgparseError(SkipPostcommandHooks):
    """
    A ``SkipPostcommandHooks`` exception for when a command fails to parse its arguments.
    Normally argparse raises a SystemExit exception in these cases. To avoid stopping the command
    loop, catch the SystemExit and raise this instead. If you still need to run post command hooks
    after parsing fails, just return instead of raising an exception.
    """
    pass