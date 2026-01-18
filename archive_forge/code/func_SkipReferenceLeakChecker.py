import copyreg
import gc
import sys
import unittest
def SkipReferenceLeakChecker(reason):
    del reason

    def Same(func):
        return func
    return Same