import warnings
from . import exceptions as exc
from . import misc
from . import normalizers
from . import validators
Create a copy of this reference with the new components.

        :param str scheme:
            (optional) The scheme to use for the new reference.
        :param str authority:
            (optional) The authority to use for the new reference.
        :param str path:
            (optional) The path to use for the new reference.
        :param str query:
            (optional) The query to use for the new reference.
        :param str fragment:
            (optional) The fragment to use for the new reference.
        :returns:
            New URIReference with provided components.
        :rtype:
            URIReference
        