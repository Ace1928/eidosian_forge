import warnings
from types import MappingProxyType
from typing import Any
from itemadapter._imports import attr, pydantic
def _get_pydantic_model_metadata(item_model: Any, field_name: str) -> MappingProxyType:
    metadata = {}
    field = item_model.__fields__[field_name].field_info
    for attribute in ['alias', 'title', 'description', 'const', 'gt', 'ge', 'lt', 'le', 'multiple_of', 'min_items', 'max_items', 'min_length', 'max_length', 'regex']:
        value = getattr(field, attribute)
        if value is not None:
            metadata[attribute] = value
    if not field.allow_mutation:
        metadata['allow_mutation'] = field.allow_mutation
    metadata.update(field.extra)
    return MappingProxyType(metadata)