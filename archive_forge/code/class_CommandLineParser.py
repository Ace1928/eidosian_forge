import os
import shlex
import subprocess
class CommandLineParser:
    """
    An object that knows how to split and join command-line arguments.

    It must be true that ``argv == split(join(argv))`` for all ``argv``.
    The reverse neednt be true - `join(split(cmd))` may result in the addition
    or removal of unnecessary escaping.
    """

    @staticmethod
    def join(argv):
        """ Join a list of arguments into a command line string """
        raise NotImplementedError

    @staticmethod
    def split(cmd):
        """ Split a command line string into a list of arguments """
        raise NotImplementedError