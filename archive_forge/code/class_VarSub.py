from __future__ import print_function, unicode_literals
import argparse
import collections
import io
import json
import logging
import os
import pprint
import re
import sys
import cmakelang
from cmakelang.format import __main__
from cmakelang import configuration
from cmakelang import lex
from cmakelang import parse
from cmakelang.parse.body_nodes import BodyNode
from cmakelang.parse.common import TreeNode
from cmakelang.parse.statement_node import StatementNode
from cmakelang.parse.funs.set import SetFnNode
class VarSub(object):

    def __init__(self, variables):
        self._vars = variables

    def __call__(self, match):
        return self._vars.get(match.group(1), '')