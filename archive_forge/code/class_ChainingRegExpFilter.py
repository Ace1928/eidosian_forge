import os
import re
import shutil
import sys
class ChainingRegExpFilter(ChainingFilter):
    """Command filter doing regexp matching for prefix commands.

    Remaining arguments are filtered again. This means that the command
    specified as the arguments must be also allowed to execute directly.
    """

    def match(self, userargs):
        if not userargs or len(self.args) > len(userargs):
            return False
        for pattern, arg in zip(self.args, userargs):
            try:
                if not re.match(pattern + '$', arg):
                    return False
            except re.error:
                return False
        return True

    def exec_args(self, userargs):
        args = userargs[len(self.args):]
        if args:
            args[0] = os.path.basename(args[0])
        return args