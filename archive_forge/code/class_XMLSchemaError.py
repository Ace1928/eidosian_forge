import sys
from xmlschema import XMLSchema as _XMLSchema
from xmlschema.exceptions import XMLSchemaException as _XMLSchemaException
import saml2.data.schemas as _data_schemas
class XMLSchemaError(Exception):
    """Generic error raised when the schema does not validate with a document"""