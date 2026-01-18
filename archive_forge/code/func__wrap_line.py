import math
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt
def _wrap_line(self, line):
    length = len(line.rstrip('\n'))
    space = '     ' if self.linenos else ''
    newline = ''
    if length > self.wrap:
        for i in range(0, math.floor(length / self.wrap)):
            chunk = line[i * self.wrap:i * self.wrap + self.wrap]
            newline += chunk + '\n' + space
        remainder = length % self.wrap
        if remainder > 0:
            newline += line[-remainder - 1:]
            self._linelen = remainder
    elif self._linelen + length > self.wrap:
        newline = '\n' + space + line
        self._linelen = length
    else:
        newline = line
        self._linelen += length
    return newline