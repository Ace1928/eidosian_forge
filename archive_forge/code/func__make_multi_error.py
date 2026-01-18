import re
from copy import deepcopy
from typing import Any, Callable, List, Match, Optional, Pattern, Tuple, Union
from docutils import nodes
from docutils.nodes import TextElement
from sphinx import addnodes
from sphinx.config import Config
from sphinx.util import logging
def _make_multi_error(self, errors: List[Any], header: str) -> DefinitionError:
    if len(errors) == 1:
        if len(header) > 0:
            return DefinitionError(header + '\n' + str(errors[0][0]))
        else:
            return DefinitionError(str(errors[0][0]))
    result = [header, '\n']
    for e in errors:
        if len(e[1]) > 0:
            indent = '  '
            result.append(e[1])
            result.append(':\n')
            for line in str(e[0]).split('\n'):
                if len(line) == 0:
                    continue
                result.append(indent)
                result.append(line)
                result.append('\n')
        else:
            result.append(str(e[0]))
    return DefinitionError(''.join(result))