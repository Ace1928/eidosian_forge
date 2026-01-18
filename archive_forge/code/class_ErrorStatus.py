from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1_modules import rfc1155
class ErrorStatus(univ.Integer):
    namedValues = namedval.NamedValues(('noError', 0), ('tooBig', 1), ('noSuchName', 2), ('badValue', 3), ('readOnly', 4), ('genErr', 5))