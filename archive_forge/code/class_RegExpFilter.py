import os
import re
import shutil
import sys
class RegExpFilter(CommandFilter):
    """Command filter doing regexp matching for every argument."""

    def match(self, userargs):
        if not userargs or len(self.args) != len(userargs):
            return False
        for pattern, arg in zip(self.args, userargs):
            try:
                if not re.match(pattern + '$', arg):
                    return False
            except re.error:
                return False
        return True