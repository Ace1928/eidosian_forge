from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions as exception
class TextDeserializer(ActionDispatcher):
    """Default request body deserialization."""

    def deserialize(self, datastring, action='default'):
        return self.dispatch(datastring, action=action)

    def default(self, datastring):
        return {}