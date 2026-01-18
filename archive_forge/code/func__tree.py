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
def _tree(self, options):
    opts = CharBuffer(options)
    if opts.peek_nonwhitespace() == '*':
        dummy = opts.next_nonwhitespace()
    name = opts.next_word()
    if opts.next_nonwhitespace() != '=':
        raise NexusError(f'Syntax error in tree description: {options[:50]}')
    rooted = False
    weight = 1.0
    while opts.peek_nonwhitespace() == '[':
        opts.next_nonwhitespace()
        symbol = next(opts)
        if symbol != '&':
            raise NexusError('Illegal special comment [%s...] in tree description: %s' % (symbol, options[:50]))
        special = next(opts)
        value = opts.next_until(']')
        next(opts)
        if special == 'R':
            rooted = True
        elif special == 'U':
            rooted = False
        elif special == 'W':
            weight = float(value)
    tree = Tree(name=name, weight=weight, rooted=rooted, tree=opts.rest().strip())
    if self.translate:
        for n in tree.get_terminals():
            try:
                tree.node(n).data.taxon = safename(self.translate[int(tree.node(n).data.taxon)])
            except (ValueError, KeyError):
                raise NexusError("Unable to substitute %s using 'translate' data." % tree.node(n).data.taxon) from None
    self.trees.append(tree)