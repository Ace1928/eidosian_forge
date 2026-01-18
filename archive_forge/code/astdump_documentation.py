import ast
import tokenize
from argparse import ArgumentParser, RawTextHelpFormatter
from enum import Enum
from nodedump import debug_format_node
Tool to dump a Python AST