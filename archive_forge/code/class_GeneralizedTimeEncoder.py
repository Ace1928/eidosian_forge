from pyasn1 import error
from pyasn1.codec.ber import encoder
from pyasn1.compat.octets import str2octs, null
from pyasn1.type import univ
from pyasn1.type import useful
class GeneralizedTimeEncoder(TimeEncoderMixIn, encoder.OctetStringEncoder):
    minLength = 12
    maxLength = 19