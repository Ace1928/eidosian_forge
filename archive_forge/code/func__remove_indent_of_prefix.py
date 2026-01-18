import difflib
from pathlib import Path
from typing import Dict, Iterable, Tuple
from parso import split_lines
from jedi.api.exceptions import RefactoringError
from jedi.inference.value.namespace import ImplicitNSName
def _remove_indent_of_prefix(prefix):
    """
    Removes the last indentation of a prefix, e.g. " \\n \\n " becomes " \\n \\n".
    """
    return ''.join(split_lines(prefix, keepends=True)[:-1])