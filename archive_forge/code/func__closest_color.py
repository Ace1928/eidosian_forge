import sys
from pygments.formatter import Formatter
from pygments.console import codes
from pygments.style import ansicolors
def _closest_color(self, r, g, b):
    distance = 257 * 257 * 3
    match = 0
    for i in range(0, 254):
        values = self.xterm_colors[i]
        rd = r - values[0]
        gd = g - values[1]
        bd = b - values[2]
        d = rd * rd + gd * gd + bd * bd
        if d < distance:
            match = i
            distance = d
    return match