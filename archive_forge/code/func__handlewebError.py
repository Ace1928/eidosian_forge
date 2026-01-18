from your HTTP server.
import pprint
import re
import socket
import sys
import time
import traceback
import os
import json
import unittest  # pylint: disable=deprecated-module,preferred-module
import warnings
import functools
import http.client
import urllib.parse
from more_itertools.more import always_iterable
import jaraco.functools
def _handlewebError(self, msg):
    print('')
    print('    ERROR: %s' % msg)
    if not self.interactive:
        raise self.failureException(msg)
    p = '    Show: [B]ody [H]eaders [S]tatus [U]RL; [I]gnore, [R]aise, or sys.e[X]it >> '
    sys.stdout.write(p)
    sys.stdout.flush()
    while True:
        i = getchar().upper()
        if not isinstance(i, type('')):
            i = i.decode('ascii')
        if i not in 'BHSUIRX':
            continue
        print(i.upper())
        if i == 'B':
            for x, line in enumerate(self.body.splitlines()):
                if (x + 1) % self.console_height == 0:
                    sys.stdout.write('<-- More -->\r')
                    m = getchar().lower()
                    sys.stdout.write('            \r')
                    if m == 'q':
                        break
                print(line)
        elif i == 'H':
            pprint.pprint(self.headers)
        elif i == 'S':
            print(self.status)
        elif i == 'U':
            print(self.url)
        elif i == 'I':
            return
        elif i == 'R':
            raise self.failureException(msg)
        elif i == 'X':
            sys.exit()
        sys.stdout.write(p)
        sys.stdout.flush()