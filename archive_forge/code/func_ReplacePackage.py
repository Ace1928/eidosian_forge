import dis
import importlib._bootstrap_external
import importlib.machinery
import marshal
import os
import io
import sys
def ReplacePackage(oldname, newname):
    replacePackageMap[oldname] = newname