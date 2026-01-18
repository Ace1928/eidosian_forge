import sys, re, curl, exceptions
from the command line first, then standard input.
class LinksysError(exceptions.Exception):

    def __init__(self, *args):
        self.args = args