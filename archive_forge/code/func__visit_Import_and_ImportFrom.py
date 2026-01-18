from ast import (
import ast
import copy
from typing import Dict, Optional, Union
def _visit_Import_and_ImportFrom(self, node: Union[Import, ImportFrom]):
    for alias in node.names:
        asname = alias.name if alias.asname is None else alias.asname
        if self.predicate(asname):
            new_name: str = 'mangle-' + asname
            self.log('Mangling Alias', new_name)
            alias.asname = new_name
        else:
            self.log('Not mangling Alias', alias.asname)
    return node