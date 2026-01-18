import base64
import typing as t
from ...filter.ldap_converters import as_guid, as_sid
from .client import SyncLDAPClient
class LDAPSchema:

    def __init__(self, attribute_types: t.Dict[str, 'sansldap.schema.AttributeTypeDescription']) -> None:
        self.attribute_types = attribute_types

    @classmethod
    def load_schema(cls, client: SyncLDAPClient) -> 'LDAPSchema':
        root_dse = client.root_dse
        attribute_types = list(client.search(filter=sansldap.FilterPresent('objectClass'), attributes=['attributeTypes'], search_base=root_dse.subschema_subentry, search_scope=sansldap.SearchScope.BASE).values())[0]['attributeTypes']
        attribute_info: t.Dict[str, sansldap.schema.AttributeTypeDescription] = {}
        for info in attribute_types:
            type_description = sansldap.schema.AttributeTypeDescription.from_string(info.decode('utf-8'))
            if type_description.names:
                attribute_info[type_description.names[0].lower()] = type_description
        return LDAPSchema(attribute_info)

    def cast_object(self, attribute: str, values: t.List[bytes]) -> t.Any:
        info = self.attribute_types.get(attribute.lower(), None)
        caster: t.Callable[[bytes], t.Any]
        if attribute == 'objectSid':
            caster = as_sid
        elif attribute == 'objectGuid':
            caster = as_guid
        elif not info or not info.syntax:
            caster = _as_str
        elif info.syntax == '1.3.6.1.4.1.1466.115.121.1.7':
            caster = _as_bool
        elif info.syntax in ['1.3.6.1.4.1.1466.115.121.1.27', '1.2.840.113556.1.4.906']:
            caster = _as_int
        elif info.syntax in ['1.3.6.1.4.1.1466.115.121.1.40', '1.2.840.113556.1.4.907', 'OctetString']:
            caster = _as_bytes
        else:
            caster = _as_str
        casted_values: t.List = []
        for v in values:
            casted_values.append(caster(v))
        if info and info.single_value:
            return casted_values[0] if casted_values else None
        else:
            return casted_values