from functools import reduce
import copy
import math
import random
import sys
import warnings
from Bio import File
from Bio.Data import IUPACData
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning, BiopythonWarning
from Bio.Nexus.StandardData import StandardData
from Bio.Nexus.Trees import Tree
class Commandline:
    """Represent a commandline as command and options."""

    def __init__(self, line, title):
        """Initialize the class."""
        self.options = {}
        options = []
        self.command = None
        try:
            self.command, options = line.strip().split('\n', 1)
        except ValueError:
            self.command = line.split()[0]
            options = ' '.join(line.split()[1:])
        self.command = self.command.strip().lower()
        if self.command in SPECIAL_COMMANDS:
            self.options = options.strip()
        elif len(options) > 0:
            try:
                options = options.replace('=', ' = ').split()
                valued_indices = [(n - 1, n, n + 1) for n in range(len(options)) if options[n] == '=' and n != 0 and (n != len(options))]
                indices = []
                for sl in valued_indices:
                    indices.extend(sl)
                token_indices = [n for n in range(len(options)) if n not in indices]
                for opt in valued_indices:
                    self.options[options[opt[0]].lower()] = options[opt[2]]
                for token in token_indices:
                    self.options[options[token].lower()] = None
            except ValueError:
                raise NexusError(f'Incorrect formatting in line: {line}') from None