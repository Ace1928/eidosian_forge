from __future__ import annotations
from copy import deepcopy
from typing import (
from bson.codec_options import DEFAULT_CODEC_OPTIONS, CodecOptions
from bson.dbref import DBRef
from bson.son import SON
from bson.timestamp import Timestamp
from pymongo import _csot, common
from pymongo.aggregation import _DatabaseAggregationCommand
from pymongo.change_stream import DatabaseChangeStream
from pymongo.collection import Collection
from pymongo.command_cursor import CommandCursor
from pymongo.common import _ecoc_coll_name, _esc_coll_name
from pymongo.errors import CollectionInvalid, InvalidName, InvalidOperation
from pymongo.read_preferences import ReadPreference, _ServerMode
from pymongo.typings import _CollationIn, _DocumentType, _DocumentTypeArg, _Pipeline
def _get_encrypted_fields(self, kwargs: Mapping[str, Any], coll_name: str, ask_db: bool) -> Optional[Mapping[str, Any]]:
    encrypted_fields = kwargs.get('encryptedFields')
    if encrypted_fields:
        return cast(Mapping[str, Any], deepcopy(encrypted_fields))
    if self.client.options.auto_encryption_opts and self.client.options.auto_encryption_opts._encrypted_fields_map and self.client.options.auto_encryption_opts._encrypted_fields_map.get(f'{self.name}.{coll_name}'):
        return cast(Mapping[str, Any], deepcopy(self.client.options.auto_encryption_opts._encrypted_fields_map[f'{self.name}.{coll_name}']))
    if ask_db and self.client.options.auto_encryption_opts:
        options = self[coll_name].options()
        if options.get('encryptedFields'):
            return cast(Mapping[str, Any], deepcopy(options['encryptedFields']))
    return None