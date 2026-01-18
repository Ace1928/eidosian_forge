from pythran.passmanager import Transformation
from pythran.tables import MODULES, pythran_ward
from pythran.syntax import PythranSyntaxError
import gast as ast
import logging
import os
def add_filename_field(node, filename):
    for descendant in ast.walk(node):
        descendant.filename = filename