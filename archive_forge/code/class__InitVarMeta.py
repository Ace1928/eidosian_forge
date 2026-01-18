import re
import sys
import copy
import types
import inspect
import keyword
class _InitVarMeta(type):

    def __getitem__(self, params):
        return self