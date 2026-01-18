from __future__ import absolute_import
import binascii
import codecs
import os
from io import BytesIO
from .fields import RequestField
from .packages import six
from .packages.six import b

    Encode a dictionary of ``fields`` using the multipart/form-data MIME format.

    :param fields:
        Dictionary of fields or list of (key, :class:`~urllib3.fields.RequestField`).

    :param boundary:
        If not specified, then a random boundary will be generated using
        :func:`urllib3.filepost.choose_boundary`.
    