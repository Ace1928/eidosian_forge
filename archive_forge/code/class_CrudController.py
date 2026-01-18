import abc
from stevedore import extension
from urllib import parse
from oslo_serialization import jsonutils
from designateclient import exceptions
class CrudController(Controller, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def list(self, *args, **kw):
        """
        List a resource
        """

    @abc.abstractmethod
    def get(self, *args, **kw):
        """
        Get a resource
        """

    @abc.abstractmethod
    def create(self, *args, **kw):
        """
        Create a resource
        """

    @abc.abstractmethod
    def update(self, *args, **kw):
        """
        Update a resource
        """

    @abc.abstractmethod
    def delete(self, *args, **kw):
        """
        Delete a resource
            """