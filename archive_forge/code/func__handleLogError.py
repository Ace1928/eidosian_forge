import sys
import time
from uuid import UUID
import pytest
from cherrypy._cpcompat import text_or_bytes
def _handleLogError(self, msg, data, marker, pattern):
    print('')
    print('    ERROR: %s' % msg)
    if not self.interactive:
        raise pytest.fail(msg)
    p = '    Show: [L]og [M]arker [P]attern; [I]gnore, [R]aise, or sys.e[X]it >> '
    sys.stdout.write(p + ' ')
    sys.stdout.flush()
    while True:
        i = getchar().upper()
        if i not in 'MPLIRX':
            continue
        print(i.upper())
        if i == 'L':
            for x, line in enumerate(data):
                if (x + 1) % self.console_height == 0:
                    sys.stdout.write('<-- More -->\r ')
                    m = getchar().lower()
                    sys.stdout.write('            \r ')
                    if m == 'q':
                        break
                print(line.rstrip())
        elif i == 'M':
            print(repr(marker or self.lastmarker))
        elif i == 'P':
            print(repr(pattern))
        elif i == 'I':
            return
        elif i == 'R':
            raise pytest.fail(msg)
        elif i == 'X':
            self.exit()
        sys.stdout.write(p + ' ')