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
class _EncryptionIO(MongoCryptCallback):

    def __init__(self, client: Optional[MongoClient[_DocumentTypeArg]], key_vault_coll: Collection[_DocumentTypeArg], mongocryptd_client: Optional[MongoClient[_DocumentTypeArg]], opts: AutoEncryptionOpts):
        """Internal class to perform I/O on behalf of pymongocrypt."""
        self.client_ref: Any
        if client is not None:
            self.client_ref = weakref.ref(client)
        else:
            self.client_ref = None
        self.key_vault_coll: Optional[Collection[RawBSONDocument]] = cast(Collection[RawBSONDocument], key_vault_coll.with_options(codec_options=_KEY_VAULT_OPTS, read_concern=ReadConcern(level='majority'), write_concern=WriteConcern(w='majority')))
        self.mongocryptd_client = mongocryptd_client
        self.opts = opts
        self._spawned = False

    def kms_request(self, kms_context: MongoCryptKmsContext) -> None:
        """Complete a KMS request.

        :Parameters:
          - `kms_context`: A :class:`MongoCryptKmsContext`.

        :Returns:
          None
        """
        endpoint = kms_context.endpoint
        message = kms_context.message
        provider = kms_context.kms_provider
        ctx = self.opts._kms_ssl_contexts.get(provider)
        if ctx is None:
            ctx = get_ssl_context(None, None, None, None, False, False, False)
        connect_timeout = max(_csot.clamp_remaining(_KMS_CONNECT_TIMEOUT), 0.001)
        opts = PoolOptions(connect_timeout=connect_timeout, socket_timeout=connect_timeout, ssl_context=ctx)
        host, port = parse_host(endpoint, _HTTPS_PORT)
        try:
            conn = _configured_socket((host, port), opts)
            try:
                conn.sendall(message)
                while kms_context.bytes_needed > 0:
                    conn.settimeout(max(_csot.clamp_remaining(_KMS_CONNECT_TIMEOUT), 0))
                    data = conn.recv(kms_context.bytes_needed)
                    if not data:
                        raise OSError('KMS connection closed')
                    kms_context.feed(data)
            except BLOCKING_IO_ERRORS:
                raise socket.timeout('timed out') from None
            finally:
                conn.close()
        except (PyMongoError, MongoCryptError):
            raise
        except Exception as error:
            _raise_connection_failure((host, port), error)

    def collection_info(self, database: Database[Mapping[str, Any]], filter: bytes) -> Optional[bytes]:
        """Get the collection info for a namespace.

        The returned collection info is passed to libmongocrypt which reads
        the JSON schema.

        :Parameters:
          - `database`: The database on which to run listCollections.
          - `filter`: The filter to pass to listCollections.

        :Returns:
          The first document from the listCollections command response as BSON.
        """
        with self.client_ref()[database].list_collections(filter=RawBSONDocument(filter)) as cursor:
            for doc in cursor:
                return _dict_to_bson(doc, False, _DATA_KEY_OPTS)
            return None

    def spawn(self) -> None:
        """Spawn mongocryptd.

        Note this method is thread safe; at most one mongocryptd will start
        successfully.
        """
        self._spawned = True
        args = [self.opts._mongocryptd_spawn_path or 'mongocryptd']
        args.extend(self.opts._mongocryptd_spawn_args)
        _spawn_daemon(args)

    def mark_command(self, database: str, cmd: bytes) -> bytes:
        """Mark a command for encryption.

        :Parameters:
          - `database`: The database on which to run this command.
          - `cmd`: The BSON command to run.

        :Returns:
          The marked command response from mongocryptd.
        """
        if not self._spawned and (not self.opts._mongocryptd_bypass_spawn):
            self.spawn()
        inflated_cmd = _inflate_bson(cmd, DEFAULT_RAW_BSON_OPTIONS)
        assert self.mongocryptd_client is not None
        try:
            res = self.mongocryptd_client[database].command(inflated_cmd, codec_options=DEFAULT_RAW_BSON_OPTIONS)
        except ServerSelectionTimeoutError:
            if self.opts._mongocryptd_bypass_spawn:
                raise
            self.spawn()
            res = self.mongocryptd_client[database].command(inflated_cmd, codec_options=DEFAULT_RAW_BSON_OPTIONS)
        return res.raw

    def fetch_keys(self, filter: bytes) -> Iterator[bytes]:
        """Yields one or more keys from the key vault.

        :Parameters:
          - `filter`: The filter to pass to find.

        :Returns:
          A generator which yields the requested keys from the key vault.
        """
        assert self.key_vault_coll is not None
        with self.key_vault_coll.find(RawBSONDocument(filter)) as cursor:
            for key in cursor:
                yield key.raw

    def insert_data_key(self, data_key: bytes) -> Binary:
        """Insert a data key into the key vault.

        :Parameters:
          - `data_key`: The data key document to insert.

        :Returns:
          The _id of the inserted data key document.
        """
        raw_doc = RawBSONDocument(data_key, _KEY_VAULT_OPTS)
        data_key_id = raw_doc.get('_id')
        if not isinstance(data_key_id, Binary) or data_key_id.subtype != UUID_SUBTYPE:
            raise TypeError('data_key _id must be Binary with a UUID subtype')
        assert self.key_vault_coll is not None
        self.key_vault_coll.insert_one(raw_doc)
        return data_key_id

    def bson_encode(self, doc: MutableMapping[str, Any]) -> bytes:
        """Encode a document to BSON.

        A document can be any mapping type (like :class:`dict`).

        :Parameters:
          - `doc`: mapping type representing a document

        :Returns:
          The encoded BSON bytes.
        """
        return encode(doc)

    def close(self) -> None:
        """Release resources.

        Note it is not safe to call this method from __del__ or any GC hooks.
        """
        self.client_ref = None
        self.key_vault_coll = None
        if self.mongocryptd_client:
            self.mongocryptd_client.close()
            self.mongocryptd_client = None