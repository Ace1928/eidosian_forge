import sys
from distutils.log import *  # noqa: F403
from distutils.log import Log as old_Log
from distutils.log import _global_log
from numpy.distutils.misc_util import (red_text, default_text, cyan_text,
class Log(old_Log):

    def _log(self, level, msg, args):
        if level >= self.threshold:
            if args:
                msg = msg % _fix_args(args)
            if 0:
                if msg.startswith('copying ') and msg.find(' -> ') != -1:
                    return
                if msg.startswith('byte-compiling '):
                    return
            print(_global_color_map[level](msg))
            sys.stdout.flush()

    def good(self, msg, *args):
        """
        If we log WARN messages, log this message as a 'nice' anti-warn
        message.

        """
        if WARN >= self.threshold:
            if args:
                print(green_text(msg % _fix_args(args)))
            else:
                print(green_text(msg))
            sys.stdout.flush()