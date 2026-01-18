from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions as exception
class JSONDeserializer(TextDeserializer):

    def _from_json(self, datastring):
        try:
            return jsonutils.loads(datastring)
        except ValueError:
            msg = _('Cannot understand JSON')
            raise exception.MalformedResponseBody(reason=msg)

    def default(self, datastring):
        return {'body': self._from_json(datastring)}