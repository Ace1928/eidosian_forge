import logging
import boto3
class ResourceMeta(object):
    """
    An object containing metadata about a resource.
    """

    def __init__(self, service_name, identifiers=None, client=None, data=None, resource_model=None):
        self.service_name = service_name
        if identifiers is None:
            identifiers = []
        self.identifiers = identifiers
        self.client = client
        self.data = data
        self.resource_model = resource_model

    def __repr__(self):
        return "ResourceMeta('{0}', identifiers={1})".format(self.service_name, self.identifiers)

    def __eq__(self, other):
        if other.__class__.__name__ != self.__class__.__name__:
            return False
        return self.__dict__ == other.__dict__

    def copy(self):
        """
        Create a copy of this metadata object.
        """
        params = self.__dict__.copy()
        service_name = params.pop('service_name')
        return ResourceMeta(service_name, **params)