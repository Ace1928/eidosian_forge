from time import sleep
from base64 import b64encode
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class DimensionDataPersistenceProfile:
    """
    Each Persistence Profile declares the combination of Virtual Listener
    type and protocol with which it is
    compatible and whether or not it is compatible as a
    Fallback Persistence Profile.
    """

    def __init__(self, id, name, compatible_listeners, fallback_compatible):
        """
        Initialize an instance of :class:`DimensionDataPersistenceProfile`

        :param id: The ID of the profile
        :type  id: ``str``

        :param name: The name of the profile
        :type  name: ``str``

        :param compatible_listeners: List of compatible Virtual Listener types
        :type  compatible_listeners: ``list`` of
            :class:`DimensionDataVirtualListenerCompatibility`

        :param fallback_compatible: Is capable as a fallback profile
        :type  fallback_compatible: ``bool``
        """
        self.id = id
        self.name = name
        self.compatible_listeners = compatible_listeners
        self.fallback_compatible = fallback_compatible

    def __repr__(self):
        return '<DimensionDataPersistenceProfile: id=%s, name=%s>' % (self.id, self.name)