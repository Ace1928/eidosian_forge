from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import useful
from pyasn1.type import univ
from pyasn1_modules import rfc5280
class EnuRevReq(univ.Enumerated):
    namedValues = namedval.NamedValues(('clrCheck', 0), ('ocspCheck', 1), ('bothCheck', 2), ('eitherCheck', 3), ('noCheck', 4), ('other', 5))