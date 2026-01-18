import errno
import re
import socket
import sys
def _parsecap(line):
    lst = line.decode('ascii').split()
    return (lst[0], lst[1:])