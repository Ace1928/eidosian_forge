from pyparsing import *
from sys import stdin, argv, exit
class Enumerate(dict):
    """C enum emulation (original by Scott David Daniels)"""

    def __init__(self, names):
        for number, name in enumerate(names.split()):
            setattr(self, name, number)
            self[number] = name