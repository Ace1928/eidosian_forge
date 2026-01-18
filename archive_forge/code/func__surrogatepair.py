import re
import sys
def _surrogatepair(c):
    return (55232 + (c >> 10), 56320 + (c & 1023))