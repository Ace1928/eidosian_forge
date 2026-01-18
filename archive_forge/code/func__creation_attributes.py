import abc
import copy
from urllib import parse as urlparse
from ironicclient.common.apiclient import base
from ironicclient import exc
@property
@abc.abstractmethod
def _creation_attributes(self):
    """A list of required creation attributes for a resource type.

        """