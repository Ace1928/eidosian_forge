from __future__ import print_function
from __future__ import unicode_literals
import collections
import logging
from cmakelang.common import InternalError
from cmakelang import lex
from cmakelang.parse.util import (
from cmakelang.parse.common import (
from cmakelang.parse.simple_nodes import (
from cmakelang.parse.statement_node import (

    Consume tokens and return a flow control tree. ``IF`` statements are special
    because they have interior ``ELSIF`` and ``ELSE`` blocks, while all other
    flow control have a single body.
    