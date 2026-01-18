import logging
import boto3
class ServiceResource(object):
    """
    A base class for resources.

    :type client: botocore.client
    :param client: A low-level Botocore client instance
    """
    meta = None
    "\n    Stores metadata about this resource instance, such as the\n    ``service_name``, the low-level ``client`` and any cached ``data``\n    from when the instance was hydrated. For example::\n\n        # Get a low-level client from a resource instance\n        client = resource.meta.client\n        response = client.operation(Param='foo')\n\n        # Print the resource instance's service short name\n        print(resource.meta.service_name)\n\n    See :py:class:`ResourceMeta` for more information.\n    "

    def __init__(self, *args, **kwargs):
        self.meta = self.meta.copy()
        if kwargs.get('client') is not None:
            self.meta.client = kwargs.get('client')
        else:
            self.meta.client = boto3.client(self.meta.service_name)
        for i, value in enumerate(args):
            setattr(self, '_' + self.meta.identifiers[i], value)
        for name, value in kwargs.items():
            if name == 'client':
                continue
            if name not in self.meta.identifiers:
                raise ValueError('Unknown keyword argument: {0}'.format(name))
            setattr(self, '_' + name, value)
        for identifier in self.meta.identifiers:
            if getattr(self, identifier) is None:
                raise ValueError('Required parameter {0} not set'.format(identifier))

    def __repr__(self):
        identifiers = []
        for identifier in self.meta.identifiers:
            identifiers.append('{0}={1}'.format(identifier, repr(getattr(self, identifier))))
        return '{0}({1})'.format(self.__class__.__name__, ', '.join(identifiers))

    def __eq__(self, other):
        if other.__class__.__name__ != self.__class__.__name__:
            return False
        for identifier in self.meta.identifiers:
            if getattr(self, identifier) != getattr(other, identifier):
                return False
        return True

    def __hash__(self):
        identifiers = []
        for identifier in self.meta.identifiers:
            identifiers.append(getattr(self, identifier))
        return hash((self.__class__.__name__, tuple(identifiers)))