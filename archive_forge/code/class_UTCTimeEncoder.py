from pyasn1 import error
from pyasn1.codec.ber import encoder
from pyasn1.compat.octets import str2octs, null
from pyasn1.type import univ
from pyasn1.type import useful
class UTCTimeEncoder(TimeEncoderMixIn, encoder.OctetStringEncoder):
    minLength = 10
    maxLength = 14