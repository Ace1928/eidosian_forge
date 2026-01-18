import re
import warnings
from inspect import Parameter, Signature
from types import ModuleType
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Sequence,
from docutils.statemachine import StringList
import sphinx
from sphinx.application import Sphinx
from sphinx.config import ENUM, Config
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.environment import BuildEnvironment
from sphinx.ext.autodoc.importer import (get_class_members, get_object_members, import_module,
from sphinx.ext.autodoc.mock import ismock, mock, undecorate
from sphinx.locale import _, __
from sphinx.pycode import ModuleAnalyzer, PycodeError
from sphinx.util import inspect, logging
from sphinx.util.docstrings import prepare_docstring, separate_metadata
from sphinx.util.inspect import (evaluate_signature, getdoc, object_description, safe_getattr,
from sphinx.util.typing import OptionSpec, get_type_hints, restify
from sphinx.util.typing import stringify as stringify_typehint
class DocstringSignatureMixin:
    """
    Mixin for FunctionDocumenter and MethodDocumenter to provide the
    feature of reading the signature from the docstring.
    """
    _new_docstrings: List[List[str]] = None
    _signatures: List[str] = None

    def _find_signature(self) -> Tuple[str, str]:
        valid_names = [self.objpath[-1]]
        if isinstance(self, ClassDocumenter):
            valid_names.append('__init__')
            if hasattr(self.object, '__mro__'):
                valid_names.extend((cls.__name__ for cls in self.object.__mro__))
        docstrings = self.get_doc()
        if docstrings is None:
            return (None, None)
        self._new_docstrings = docstrings[:]
        self._signatures = []
        result = None
        for i, doclines in enumerate(docstrings):
            for j, line in enumerate(doclines):
                if not line:
                    break
                if line.endswith('\\'):
                    line = line.rstrip('\\').rstrip()
                match = py_ext_sig_re.match(line)
                if not match:
                    break
                exmod, path, base, args, retann = match.groups()
                if base not in valid_names:
                    break
                tab_width = self.directive.state.document.settings.tab_width
                self._new_docstrings[i] = prepare_docstring('\n'.join(doclines[j + 1:]), tab_width)
                if result is None:
                    result = (args, retann)
                else:
                    self._signatures.append('(%s) -> %s' % (args, retann))
            if result:
                break
        return result

    def get_doc(self) -> List[List[str]]:
        if self._new_docstrings is not None:
            return self._new_docstrings
        return super().get_doc()

    def format_signature(self, **kwargs: Any) -> str:
        if self.args is None and self.config.autodoc_docstring_signature:
            result = self._find_signature()
            if result is not None:
                self.args, self.retann = result
        sig = super().format_signature(**kwargs)
        if self._signatures:
            return '\n'.join([sig] + self._signatures)
        else:
            return sig