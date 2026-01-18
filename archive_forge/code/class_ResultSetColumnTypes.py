import sys
from collections import OrderedDict
from distutils.util import strtobool
from aiokeydb.v1.exceptions import ResponseError
from aiokeydb.v1.commands.graph.edge import Edge
from aiokeydb.v1.commands.graph.exceptions import VersionMismatchException
from aiokeydb.v1.commands.graph.node import Node
from aiokeydb.v1.commands.graph.path import Path
class ResultSetColumnTypes:
    COLUMN_UNKNOWN = 0
    COLUMN_SCALAR = 1
    COLUMN_NODE = 2
    COLUMN_RELATION = 3