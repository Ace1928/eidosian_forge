from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class ParseProtoException(exceptions.Error):
    """Error interpreting a dictionary as a specific proto message."""

    def __init__(self, path, proto_name, msg):
        msg = 'interpreting {path} as {proto_name}: {msg}'.format(path=path, proto_name=proto_name, msg=msg)
        super(ParseProtoException, self).__init__(msg)