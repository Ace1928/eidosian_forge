from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1_modules import rfc2459
class OCSPResponseStatus(univ.Enumerated):
    namedValues = namedval.NamedValues(('successful', 0), ('malformedRequest', 1), ('internalError', 2), ('tryLater', 3), ('undefinedStatus', 4), ('sigRequired', 5), ('unauthorized', 6))