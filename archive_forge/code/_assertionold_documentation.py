import py
import sys, inspect
from compiler import parse, ast, pycodegen
from py._code.assertion import BuiltinAssertionError, _format_explanation
import types
A parse tree node with a few extra methods.