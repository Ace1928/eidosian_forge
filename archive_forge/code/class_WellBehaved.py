from twisted.python import usage
from twisted.trial import unittest
class WellBehaved(usage.Options):
    optParameters = [['long', 'w', 'default', 'and a docstring'], ['another', 'n', 'no docstring'], ['longonly', None, 'noshort'], ['shortless', None, 'except', 'this one got docstring']]
    optFlags = [['aflag', 'f', '\n\n                 flagallicious docstringness for this here\n\n                 '], ['flout', 'o']]

    def opt_myflag(self):
        self.opts['myflag'] = 'PONY!'

    def opt_myparam(self, value):
        self.opts['myparam'] = f'{value} WITH A PONY!'