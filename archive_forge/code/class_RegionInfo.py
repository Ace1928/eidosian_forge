import os
import boto
from boto.compat import json
from boto.exception import BotoClientError
from boto.endpoints import BotoEndpointResolver
from boto.endpoints import StaticEndpointBuilder
class RegionInfo(object):
    """
    Represents an AWS Region
    """

    def __init__(self, connection=None, name=None, endpoint=None, connection_cls=None):
        self.connection = connection
        self.name = name
        self.endpoint = endpoint
        self.connection_cls = connection_cls

    def __repr__(self):
        return 'RegionInfo:%s' % self.name

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'regionName':
            self.name = value
        elif name == 'regionEndpoint':
            self.endpoint = value
        else:
            setattr(self, name, value)

    def connect(self, **kw_params):
        """
        Connect to this Region's endpoint. Returns an connection
        object pointing to the endpoint associated with this region.
        You may pass any of the arguments accepted by the connection
        class's constructor as keyword arguments and they will be
        passed along to the connection object.

        :rtype: Connection object
        :return: The connection to this regions endpoint
        """
        if self.connection_cls:
            return self.connection_cls(region=self, **kw_params)