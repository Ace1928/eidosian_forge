import os
import typing
import warnings
from types import ModuleType
from warnings import warn
import rpy2.rinterface as rinterface
from . import conversion
from rpy2.robjects.functions import (SignatureTranslatedFunction,
from rpy2.robjects import Environment
from rpy2.robjects.packages_utils import (
import rpy2.robjects.help as rhelp
@docstring_property(__doc__)
def __doc__(self):
    doc = list(['Python representation of an R package.', 'R arguments:', ''])
    if not self.__rname__:
        doc.append('<No information available>')
    else:
        try:
            doc.append(rhelp.docstring(self.__rname__, self.__rname__ + '-package', sections=['\\description']))
        except rhelp.HelpNotFoundError:
            doc.append('[R help was not found]')
    return os.linesep.join(doc)