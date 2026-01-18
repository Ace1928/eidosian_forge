from collections import defaultdict
import datetime
import io
import json
from prov import Error
from prov.serializers import Serializer
from prov.constants import *
from prov.model import (
import logging
class ProvJSONSerializer(Serializer):
    """
    PROV-JSON serializer for :class:`~prov.model.ProvDocument`
    """

    def serialize(self, stream, **kwargs):
        """
        Serializes a :class:`~prov.model.ProvDocument` instance to
        `PROV-JSON <https://openprovenance.org/prov-json/>`_.

        :param stream: Where to save the output.
        """
        buf = io.StringIO()
        try:
            json.dump(self.document, buf, cls=ProvJSONEncoder, **kwargs)
            buf.seek(0, 0)
            if isinstance(stream, io.TextIOBase):
                stream.write(buf.read())
            else:
                stream.write(buf.read().encode('utf-8'))
        finally:
            buf.close()

    def deserialize(self, stream, **kwargs):
        """
        Deserialize from the `PROV JSON
        <https://openprovenance.org/prov-json/>`_ representation to a
        :class:`~prov.model.ProvDocument` instance.

        :param stream: Input data.
        """
        if not isinstance(stream, io.TextIOBase):
            buf = io.StringIO(stream.read().decode('utf-8'))
            stream = buf
        return json.load(stream, cls=ProvJSONDecoder, **kwargs)