import jsonschema
from oslo_utils import encodeutils
from glance.common import exception
from glance.i18n import _
class PermissiveSchema(Schema):

    @staticmethod
    def _filter_func(properties, key):
        return True

    def raw(self):
        raw = super(PermissiveSchema, self).raw()
        raw['additionalProperties'] = {'type': 'string'}
        return raw

    def minimal(self):
        minimal = super(PermissiveSchema, self).raw()
        return minimal