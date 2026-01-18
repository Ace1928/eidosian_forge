import warnings
from twisted.trial.unittest import TestCase
class FXF(Flags):
    READ = FlagConstant()
    WRITE = FlagConstant()
    APPEND = FlagConstant()
    EXCLUSIVE = FlagConstant(32)
    TEXT = FlagConstant()