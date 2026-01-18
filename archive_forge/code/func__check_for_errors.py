import sys
from collections import OrderedDict
from distutils.util import strtobool
from aiokeydb.v1.exceptions import ResponseError
from aiokeydb.v1.commands.graph.edge import Edge
from aiokeydb.v1.commands.graph.exceptions import VersionMismatchException
from aiokeydb.v1.commands.graph.node import Node
from aiokeydb.v1.commands.graph.path import Path
def _check_for_errors(self, response):
    """
        Check if the response contains an error.
        """
    if isinstance(response[0], ResponseError):
        error = response[0]
        if str(error) == 'version mismatch':
            version = response[1]
            error = VersionMismatchException(version)
        raise error
    if isinstance(response[-1], ResponseError):
        raise response[-1]