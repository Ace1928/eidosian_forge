from twisted.python import usage
from twisted.trial import unittest
class OptBar(usage.Options):
    subCommands = [('bar', 'b', SubOpt, 'quux')]