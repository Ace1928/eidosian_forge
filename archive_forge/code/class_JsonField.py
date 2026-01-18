from oslo_serialization import jsonutils as json
from oslo_versionedobjects import fields
class JsonField(fields.AutoTypedField):
    AUTO_TYPE = Json()