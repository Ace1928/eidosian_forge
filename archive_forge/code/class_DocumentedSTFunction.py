import inspect
import os
import re
import textwrap
import typing
from typing import Union
import warnings
from collections import OrderedDict
from rpy2.robjects.robject import RObjectMixin
import rpy2.rinterface as rinterface
import rpy2.rinterface_lib.sexp
from rpy2.robjects import help
from rpy2.robjects import conversion
from rpy2.robjects.vectors import Vector
from rpy2.robjects.packages_utils import (default_symbol_r2python,
class DocumentedSTFunction(SignatureTranslatedFunction):

    def __init__(self, sexp: rinterface.SexpClosure, init_prm_translate=None, packagename: typing.Optional[str]=None):
        super(DocumentedSTFunction, self).__init__(sexp, init_prm_translate=init_prm_translate)
        self.__rpackagename__ = packagename

    @docstring_property(__doc__)
    def __doc__(self):
        package = help.Package(self.__rpackagename__)
        page = package.fetch(self.__rname__)
        doc = ['Wrapper around an R function.', '', 'The docstring below is built from the R documentation.', '']
        if '\\description' in page.sections:
            doc.append(page.to_docstring(section_names=['\\description']))
        fm = _formals_fixed(self)
        if fm is rpy2.rinterface_lib.sexp.NULL:
            names = tuple()
        else:
            names = fm.do_slot('names')
        doc.append(self.__rname__ + '(')
        for key, val in self._prm_translate.items():
            if key == '___':
                description = '(was "..."). R ellipsis (any number of parameters)'
            else:
                description = _repr_argval(fm[names.index(val)])
            if description is None:
                doc.append('    %s,' % key)
            else:
                doc.append('    %s = %s,' % (key, description))
        doc.extend((')', ''))
        doc.append('Args:')
        for item in page.arguments():
            description = ('%s  ' % os.linesep).join(item.value)
            doc.append(' '.join(('  ', item.name, ': ', description)))
            doc.append('')
        if '\\details' in page.sections:
            doc.append(page.to_docstring(section_names=['\\details']))
        return os.linesep.join(doc)