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
class SlotsMixin(DataDocumenterMixinBase):
    """
    Mixin for AttributeDocumenter to provide the feature for supporting __slots__.
    """

    def isslotsattribute(self) -> bool:
        """Check the subject is an attribute in __slots__."""
        try:
            __slots__ = inspect.getslots(self.parent)
            if __slots__ and self.objpath[-1] in __slots__:
                return True
            else:
                return False
        except (ValueError, TypeError):
            return False

    def import_object(self, raiseerror: bool=False) -> bool:
        ret = super().import_object(raiseerror)
        if self.isslotsattribute():
            self.object = SLOTSATTR
        return ret

    def should_suppress_value_header(self) -> bool:
        if self.object is SLOTSATTR:
            return True
        else:
            return super().should_suppress_value_header()

    def get_doc(self) -> Optional[List[List[str]]]:
        if self.object is SLOTSATTR:
            try:
                __slots__ = inspect.getslots(self.parent)
                if __slots__ and __slots__.get(self.objpath[-1]):
                    docstring = prepare_docstring(__slots__[self.objpath[-1]])
                    return [docstring]
                else:
                    return []
            except ValueError as exc:
                logger.warning(__('Invalid __slots__ found on %s. Ignored.'), (self.parent.__qualname__, exc), type='autodoc')
                return []
        else:
            return super().get_doc()

    @property
    def _datadescriptor(self) -> bool:
        warnings.warn('AttributeDocumenter._datadescriptor() is deprecated.', RemovedInSphinx60Warning)
        if self.object is SLOTSATTR:
            return True
        else:
            return False