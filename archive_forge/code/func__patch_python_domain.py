from typing import Any, Dict, List
import sphinx
from sphinx.application import Sphinx
from sphinx.ext.napoleon.docstring import GoogleDocstring, NumpyDocstring
from sphinx.util import inspect
def _patch_python_domain() -> None:
    try:
        from sphinx.domains.python import PyTypedField
    except ImportError:
        pass
    else:
        import sphinx.domains.python
        from sphinx.locale import _
        for doc_field in sphinx.domains.python.PyObject.doc_field_types:
            if doc_field.name == 'parameter':
                doc_field.names = ('param', 'parameter', 'arg', 'argument')
                break
        sphinx.domains.python.PyObject.doc_field_types.append(PyTypedField('keyword', label=_('Keyword Arguments'), names=('keyword', 'kwarg', 'kwparam'), typerolename='obj', typenames=('paramtype', 'kwtype'), can_collapse=True))