from collections import namedtuple
import re
import textwrap
import warnings
class AcceptOffer(namedtuple('AcceptOffer', ['type', 'subtype', 'params'])):
    """
    A pre-parsed offer tuple represeting a value in the format
    ``type/subtype;param0=value0;param1=value1``.

    :ivar type: The media type's root category.
    :ivar subtype: The media type's subtype.
    :ivar params: A tuple of 2-tuples containing parameter names and values.

    """
    __slots__ = ()

    def __str__(self):
        """
        Return the properly quoted media type string.

        """
        value = self.type + '/' + self.subtype
        return Accept._form_media_range(value, self.params)