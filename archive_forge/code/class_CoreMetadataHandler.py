from __future__ import annotations as _annotations
import typing
from typing import Any
import typing_extensions
class CoreMetadataHandler:
    """Because the metadata field in pydantic_core is of type `Any`, we can't assume much about its contents.

    This class is used to interact with the metadata field on a CoreSchema object in a consistent
    way throughout pydantic.
    """
    __slots__ = ('_schema',)

    def __init__(self, schema: CoreSchemaOrField):
        self._schema = schema
        metadata = schema.get('metadata')
        if metadata is None:
            schema['metadata'] = CoreMetadata()
        elif not isinstance(metadata, dict):
            raise TypeError(f'CoreSchema metadata should be a dict; got {metadata!r}.')

    @property
    def metadata(self) -> CoreMetadata:
        """Retrieves the metadata dict from the schema, initializing it to a dict if it is None
        and raises an error if it is not a dict.
        """
        metadata = self._schema.get('metadata')
        if metadata is None:
            self._schema['metadata'] = metadata = CoreMetadata()
        if not isinstance(metadata, dict):
            raise TypeError(f'CoreSchema metadata should be a dict; got {metadata!r}.')
        return metadata