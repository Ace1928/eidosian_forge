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
class TypeVarMixin(DataDocumenterMixinBase):
    """
    Mixin for DataDocumenter and AttributeDocumenter to provide the feature for
    supporting TypeVars.
    """

    def should_suppress_directive_header(self) -> bool:
        return isinstance(self.object, TypeVar) or super().should_suppress_directive_header()

    def get_doc(self) -> Optional[List[List[str]]]:
        if isinstance(self.object, TypeVar):
            if self.object.__doc__ != TypeVar.__doc__:
                return super().get_doc()
            else:
                return []
        else:
            return super().get_doc()

    def update_content(self, more_content: StringList) -> None:
        if isinstance(self.object, TypeVar):
            attrs = [repr(self.object.__name__)]
            for constraint in self.object.__constraints__:
                if self.config.autodoc_typehints_format == 'short':
                    attrs.append(stringify_typehint(constraint, 'smart'))
                else:
                    attrs.append(stringify_typehint(constraint))
            if self.object.__bound__:
                if self.config.autodoc_typehints_format == 'short':
                    bound = restify(self.object.__bound__, 'smart')
                else:
                    bound = restify(self.object.__bound__)
                attrs.append('bound=\\ ' + bound)
            if self.object.__covariant__:
                attrs.append('covariant=True')
            if self.object.__contravariant__:
                attrs.append('contravariant=True')
            more_content.append(_('alias of TypeVar(%s)') % ', '.join(attrs), '')
            more_content.append('', '')
        super().update_content(more_content)