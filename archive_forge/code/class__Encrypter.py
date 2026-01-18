from __future__ import annotations
import contextlib
import enum
import socket
import weakref
from copy import deepcopy
from typing import (
from bson import _dict_to_bson, decode, encode
from bson.binary import STANDARD, UUID_SUBTYPE, Binary
from bson.codec_options import CodecOptions
from bson.errors import BSONError
from bson.raw_bson import DEFAULT_RAW_BSON_OPTIONS, RawBSONDocument, _inflate_bson
from bson.son import SON
from pymongo import _csot
from pymongo.collection import Collection
from pymongo.common import CONNECT_TIMEOUT
from pymongo.cursor import Cursor
from pymongo.daemon import _spawn_daemon
from pymongo.database import Database
from pymongo.encryption_options import AutoEncryptionOpts, RangeOpts
from pymongo.errors import (
from pymongo.mongo_client import MongoClient
from pymongo.network import BLOCKING_IO_ERRORS
from pymongo.operations import UpdateOne
from pymongo.pool import PoolOptions, _configured_socket, _raise_connection_failure
from pymongo.read_concern import ReadConcern
from pymongo.results import BulkWriteResult, DeleteResult
from pymongo.ssl_support import get_ssl_context
from pymongo.typings import _DocumentType, _DocumentTypeArg
from pymongo.uri_parser import parse_host
from pymongo.write_concern import WriteConcern
class _Encrypter:
    """Encrypts and decrypts MongoDB commands.

    This class is used to support automatic encryption and decryption of
    MongoDB commands.
    """

    def __init__(self, client: MongoClient[_DocumentTypeArg], opts: AutoEncryptionOpts):
        """Create a _Encrypter for a client.

        :Parameters:
          - `client`: The encrypted MongoClient.
          - `opts`: The encrypted client's :class:`AutoEncryptionOpts`.
        """
        if opts._schema_map is None:
            schema_map = None
        else:
            schema_map = _dict_to_bson(opts._schema_map, False, _DATA_KEY_OPTS)
        if opts._encrypted_fields_map is None:
            encrypted_fields_map = None
        else:
            encrypted_fields_map = _dict_to_bson(opts._encrypted_fields_map, False, _DATA_KEY_OPTS)
        self._bypass_auto_encryption = opts._bypass_auto_encryption
        self._internal_client = None

        def _get_internal_client(encrypter: _Encrypter, mongo_client: MongoClient[_DocumentTypeArg]) -> MongoClient[_DocumentTypeArg]:
            if mongo_client.options.pool_options.max_pool_size is None:
                return mongo_client
            if encrypter._internal_client is not None:
                return encrypter._internal_client
            internal_client = mongo_client._duplicate(minPoolSize=0, auto_encryption_opts=None)
            encrypter._internal_client = internal_client
            return internal_client
        if opts._key_vault_client is not None:
            key_vault_client = opts._key_vault_client
        else:
            key_vault_client = _get_internal_client(self, client)
        if opts._bypass_auto_encryption:
            metadata_client = None
        else:
            metadata_client = _get_internal_client(self, client)
        db, coll = opts._key_vault_namespace.split('.', 1)
        key_vault_coll = key_vault_client[db][coll]
        mongocryptd_client: MongoClient[Mapping[str, Any]] = MongoClient(opts._mongocryptd_uri, connect=False, serverSelectionTimeoutMS=_MONGOCRYPTD_TIMEOUT_MS)
        io_callbacks = _EncryptionIO(metadata_client, key_vault_coll, mongocryptd_client, opts)
        self._auto_encrypter = AutoEncrypter(io_callbacks, MongoCryptOptions(opts._kms_providers, schema_map, crypt_shared_lib_path=opts._crypt_shared_lib_path, crypt_shared_lib_required=opts._crypt_shared_lib_required, bypass_encryption=opts._bypass_auto_encryption, encrypted_fields_map=encrypted_fields_map, bypass_query_analysis=opts._bypass_query_analysis))
        self._closed = False

    def encrypt(self, database: str, cmd: Mapping[str, Any], codec_options: CodecOptions[_DocumentTypeArg]) -> MutableMapping[str, Any]:
        """Encrypt a MongoDB command.

        :Parameters:
          - `database`: The database for this command.
          - `cmd`: A command document.
          - `codec_options`: The CodecOptions to use while encoding `cmd`.

        :Returns:
          The encrypted command to execute.
        """
        self._check_closed()
        encoded_cmd = _dict_to_bson(cmd, False, codec_options)
        with _wrap_encryption_errors():
            encrypted_cmd = self._auto_encrypter.encrypt(database, encoded_cmd)
            return _inflate_bson(encrypted_cmd, DEFAULT_RAW_BSON_OPTIONS)

    def decrypt(self, response: bytes) -> Optional[bytes]:
        """Decrypt a MongoDB command response.

        :Parameters:
          - `response`: A MongoDB command response as BSON.

        :Returns:
          The decrypted command response.
        """
        self._check_closed()
        with _wrap_encryption_errors():
            return cast(bytes, self._auto_encrypter.decrypt(response))

    def _check_closed(self) -> None:
        if self._closed:
            raise InvalidOperation('Cannot use MongoClient after close')

    def close(self) -> None:
        """Cleanup resources."""
        self._closed = True
        self._auto_encrypter.close()
        if self._internal_client:
            self._internal_client.close()
            self._internal_client = None