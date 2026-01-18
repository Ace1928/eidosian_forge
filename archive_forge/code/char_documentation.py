import sys
from pyasn1 import error
from pyasn1.type import tag
from pyasn1.type import univ
Creates |ASN.1| schema or value object.

    |ASN.1| objects are immutable and duck-type Python 2 :class:`unicode` or Python 3 :class:`str`.
    When used in octet-stream context, |ASN.1| type assumes "|encoding|" encoding.

    Keyword Args
    ------------
    value: :class:`unicode`, :class:`str`, :class:`bytes` or |ASN.1| object
        unicode object (Python 2) or string (Python 3), alternatively string
        (Python 2) or bytes (Python 3) representing octet-stream of serialised
        unicode string (note `encoding` parameter) or |ASN.1| class instance.

    tagSet: :py:class:`~pyasn1.type.tag.TagSet`
        Object representing non-default ASN.1 tag(s)

    subtypeSpec: :py:class:`~pyasn1.type.constraint.ConstraintsIntersection`
        Object representing non-default ASN.1 subtype constraint(s)

    encoding: :py:class:`str`
        Unicode codec ID to encode/decode :class:`unicode` (Python 2) or
        :class:`str` (Python 3) the payload when |ASN.1| object is used
        in octet-stream context.

    Raises
    ------
    :py:class:`~pyasn1.error.PyAsn1Error`
        On constraint violation or bad initializer.
    