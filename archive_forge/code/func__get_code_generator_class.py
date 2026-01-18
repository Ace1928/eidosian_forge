from functools import partial, update_wrapper
from .draft04 import CodeGeneratorDraft04
from .draft06 import CodeGeneratorDraft06
from .draft07 import CodeGeneratorDraft07
from .exceptions import JsonSchemaException, JsonSchemaValueException, JsonSchemaDefinitionException
from .ref_resolver import RefResolver
from .version import VERSION
def _get_code_generator_class(schema):
    if isinstance(schema, dict):
        schema_version = schema.get('$schema', '')
        if 'draft-04' in schema_version:
            return CodeGeneratorDraft04
        if 'draft-06' in schema_version:
            return CodeGeneratorDraft06
    return CodeGeneratorDraft07