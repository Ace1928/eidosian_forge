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
class ClientEncryption(Generic[_DocumentType]):
    """Explicit client-side field level encryption."""

    def __init__(self, kms_providers: Mapping[str, Any], key_vault_namespace: str, key_vault_client: MongoClient[_DocumentTypeArg], codec_options: CodecOptions[_DocumentTypeArg], kms_tls_options: Optional[Mapping[str, Any]]=None) -> None:
        """Explicit client-side field level encryption.

        The ClientEncryption class encapsulates explicit operations on a key
        vault collection that cannot be done directly on a MongoClient. Similar
        to configuring auto encryption on a MongoClient, it is constructed with
        a MongoClient (to a MongoDB cluster containing the key vault
        collection), KMS provider configuration, and keyVaultNamespace. It
        provides an API for explicitly encrypting and decrypting values, and
        creating data keys. It does not provide an API to query keys from the
        key vault collection, as this can be done directly on the MongoClient.

        See :ref:`explicit-client-side-encryption` for an example.

        :Parameters:
          - `kms_providers`: Map of KMS provider options. The `kms_providers`
            map values differ by provider:

              - `aws`: Map with "accessKeyId" and "secretAccessKey" as strings.
                These are the AWS access key ID and AWS secret access key used
                to generate KMS messages. An optional "sessionToken" may be
                included to support temporary AWS credentials.
              - `azure`: Map with "tenantId", "clientId", and "clientSecret" as
                strings. Additionally, "identityPlatformEndpoint" may also be
                specified as a string (defaults to 'login.microsoftonline.com').
                These are the Azure Active Directory credentials used to
                generate Azure Key Vault messages.
              - `gcp`: Map with "email" as a string and "privateKey"
                as `bytes` or a base64 encoded string.
                Additionally, "endpoint" may also be specified as a string
                (defaults to 'oauth2.googleapis.com'). These are the
                credentials used to generate Google Cloud KMS messages.
              - `kmip`: Map with "endpoint" as a host with required port.
                For example: ``{"endpoint": "example.com:443"}``.
              - `local`: Map with "key" as `bytes` (96 bytes in length) or
                a base64 encoded string which decodes
                to 96 bytes. "key" is the master key used to encrypt/decrypt
                data keys. This key should be generated and stored as securely
                as possible.

          - `key_vault_namespace`: The namespace for the key vault collection.
            The key vault collection contains all data keys used for encryption
            and decryption. Data keys are stored as documents in this MongoDB
            collection. Data keys are protected with encryption by a KMS
            provider.
          - `key_vault_client`: A MongoClient connected to a MongoDB cluster
            containing the `key_vault_namespace` collection.
          - `codec_options`: An instance of
            :class:`~bson.codec_options.CodecOptions` to use when encoding a
            value for encryption and decoding the decrypted BSON value. This
            should be the same CodecOptions instance configured on the
            MongoClient, Database, or Collection used to access application
            data.
          - `kms_tls_options` (optional): A map of KMS provider names to TLS
            options to use when creating secure connections to KMS providers.
            Accepts the same TLS options as
            :class:`pymongo.mongo_client.MongoClient`. For example, to
            override the system default CA file::

              kms_tls_options={'kmip': {'tlsCAFile': certifi.where()}}

            Or to supply a client certificate::

              kms_tls_options={'kmip': {'tlsCertificateKeyFile': 'client.pem'}}

        .. versionchanged:: 4.0
           Added the `kms_tls_options` parameter and the "kmip" KMS provider.

        .. versionadded:: 3.9
        """
        if not _HAVE_PYMONGOCRYPT:
            raise ConfigurationError("client-side field level encryption requires the pymongocrypt library: install a compatible version with: python -m pip install 'pymongo[encryption]'")
        if not isinstance(codec_options, CodecOptions):
            raise TypeError('codec_options must be an instance of bson.codec_options.CodecOptions')
        self._kms_providers = kms_providers
        self._key_vault_namespace = key_vault_namespace
        self._key_vault_client = key_vault_client
        self._codec_options = codec_options
        db, coll = key_vault_namespace.split('.', 1)
        key_vault_coll = key_vault_client[db][coll]
        opts = AutoEncryptionOpts(kms_providers, key_vault_namespace, kms_tls_options=kms_tls_options)
        self._io_callbacks: Optional[_EncryptionIO] = _EncryptionIO(None, key_vault_coll, None, opts)
        self._encryption = ExplicitEncrypter(self._io_callbacks, MongoCryptOptions(kms_providers, None))
        assert self._io_callbacks.key_vault_coll is not None
        self._key_vault_coll = self._io_callbacks.key_vault_coll

    def create_encrypted_collection(self, database: Database[_DocumentTypeArg], name: str, encrypted_fields: Mapping[str, Any], kms_provider: Optional[str]=None, master_key: Optional[Mapping[str, Any]]=None, **kwargs: Any) -> tuple[Collection[_DocumentTypeArg], Mapping[str, Any]]:
        """Create a collection with encryptedFields.

        .. warning::
            This function does not update the encryptedFieldsMap in the client's
            AutoEncryptionOpts, thus the user must create a new client after calling this function with
            the encryptedFields returned.

        Normally collection creation is automatic. This method should
        only be used to specify options on
        creation. :class:`~pymongo.errors.EncryptionError` will be
        raised if the collection already exists.

        :Parameters:
          - `name`: the name of the collection to create
          - `encrypted_fields` (dict): Document that describes the encrypted fields for
            Queryable Encryption. For example::

              {
                "escCollection": "enxcol_.encryptedCollection.esc",
                "ecocCollection": "enxcol_.encryptedCollection.ecoc",
                "fields": [
                    {
                        "path": "firstName",
                        "keyId": Binary.from_uuid(UUID('00000000-0000-0000-0000-000000000000')),
                        "bsonType": "string",
                        "queries": {"queryType": "equality"}
                    },
                    {
                        "path": "ssn",
                        "keyId": Binary.from_uuid(UUID('04104104-1041-0410-4104-104104104104')),
                        "bsonType": "string"
                    }
                  ]
              }

            The "keyId" may be set to ``None`` to auto-generate the data keys.
          - `kms_provider` (optional): the KMS provider to be used
          - `master_key` (optional): Identifies a KMS-specific key used to encrypt the
            new data key. If the kmsProvider is "local" the `master_key` is
            not applicable and may be omitted.
          - `**kwargs` (optional): additional keyword arguments are the same as "create_collection".

        All optional `create collection command`_ parameters should be passed
        as keyword arguments to this method.
        See the documentation for :meth:`~pymongo.database.Database.create_collection` for all valid options.

        :Raises:
          - :class:`~pymongo.errors.EncryptedCollectionError`: When either data-key creation or creating the collection fails.

        .. versionadded:: 4.4

        .. _create collection command:
            https://mongodb.com/docs/manual/reference/command/create

        """
        encrypted_fields = deepcopy(encrypted_fields)
        for i, field in enumerate(encrypted_fields['fields']):
            if isinstance(field, dict) and field.get('keyId') is None:
                try:
                    encrypted_fields['fields'][i]['keyId'] = self.create_data_key(kms_provider=kms_provider, master_key=master_key)
                except EncryptionError as exc:
                    raise EncryptedCollectionError(exc, encrypted_fields) from exc
        kwargs['encryptedFields'] = encrypted_fields
        kwargs['check_exists'] = False
        try:
            return (database.create_collection(name=name, **kwargs), encrypted_fields)
        except Exception as exc:
            raise EncryptedCollectionError(exc, encrypted_fields) from exc

    def create_data_key(self, kms_provider: str, master_key: Optional[Mapping[str, Any]]=None, key_alt_names: Optional[Sequence[str]]=None, key_material: Optional[bytes]=None) -> Binary:
        """Create and insert a new data key into the key vault collection.

        :Parameters:
          - `kms_provider`: The KMS provider to use. Supported values are
            "aws", "azure", "gcp", "kmip", and "local".
          - `master_key`: Identifies a KMS-specific key used to encrypt the
            new data key. If the kmsProvider is "local" the `master_key` is
            not applicable and may be omitted.

            If the `kms_provider` is "aws" it is required and has the
            following fields::

              - `region` (string): Required. The AWS region, e.g. "us-east-1".
              - `key` (string): Required. The Amazon Resource Name (ARN) to
                 the AWS customer.
              - `endpoint` (string): Optional. An alternate host to send KMS
                requests to. May include port number, e.g.
                "kms.us-east-1.amazonaws.com:443".

            If the `kms_provider` is "azure" it is required and has the
            following fields::

              - `keyVaultEndpoint` (string): Required. Host with optional
                 port, e.g. "example.vault.azure.net".
              - `keyName` (string): Required. Key name in the key vault.
              - `keyVersion` (string): Optional. Version of the key to use.

            If the `kms_provider` is "gcp" it is required and has the
            following fields::

              - `projectId` (string): Required. The Google cloud project ID.
              - `location` (string): Required. The GCP location, e.g. "us-east1".
              - `keyRing` (string): Required. Name of the key ring that contains
                the key to use.
              - `keyName` (string): Required. Name of the key to use.
              - `keyVersion` (string): Optional. Version of the key to use.
              - `endpoint` (string): Optional. Host with optional port.
                Defaults to "cloudkms.googleapis.com".

            If the `kms_provider` is "kmip" it is optional and has the
            following fields::

              - `keyId` (string): Optional. `keyId` is the KMIP Unique
                Identifier to a 96 byte KMIP Secret Data managed object. If
                keyId is omitted, the driver creates a random 96 byte KMIP
                Secret Data managed object.
              - `endpoint` (string): Optional. Host with optional
                 port, e.g. "example.vault.azure.net:".

          - `key_alt_names` (optional): An optional list of string alternate
            names used to reference a key. If a key is created with alternate
            names, then encryption may refer to the key by the unique alternate
            name instead of by ``key_id``. The following example shows creating
            and referring to a data key by alternate name::

              client_encryption.create_data_key("local", key_alt_names=["name1"])
              # reference the key with the alternate name
              client_encryption.encrypt("457-55-5462", key_alt_name="name1",
                                        algorithm=Algorithm.AEAD_AES_256_CBC_HMAC_SHA_512_Random)
          - `key_material` (optional): Sets the custom key material to be used
            by the data key for encryption and decryption.

        :Returns:
          The ``_id`` of the created data key document as a
          :class:`~bson.binary.Binary` with subtype
          :data:`~bson.binary.UUID_SUBTYPE`.

        .. versionchanged:: 4.2
           Added the `key_material` parameter.
        """
        self._check_closed()
        with _wrap_encryption_errors():
            return cast(Binary, self._encryption.create_data_key(kms_provider, master_key=master_key, key_alt_names=key_alt_names, key_material=key_material))

    def _encrypt_helper(self, value: Any, algorithm: str, key_id: Optional[Binary]=None, key_alt_name: Optional[str]=None, query_type: Optional[str]=None, contention_factor: Optional[int]=None, range_opts: Optional[RangeOpts]=None, is_expression: bool=False) -> Any:
        self._check_closed()
        if key_id is not None and (not (isinstance(key_id, Binary) and key_id.subtype == UUID_SUBTYPE)):
            raise TypeError('key_id must be a bson.binary.Binary with subtype 4')
        doc = encode({'v': value}, codec_options=self._codec_options)
        range_opts_bytes = None
        if range_opts:
            range_opts_bytes = encode(range_opts.document, codec_options=self._codec_options)
        with _wrap_encryption_errors():
            encrypted_doc = self._encryption.encrypt(value=doc, algorithm=algorithm, key_id=key_id, key_alt_name=key_alt_name, query_type=query_type, contention_factor=contention_factor, range_opts=range_opts_bytes, is_expression=is_expression)
            return decode(encrypted_doc)['v']

    def encrypt(self, value: Any, algorithm: str, key_id: Optional[Binary]=None, key_alt_name: Optional[str]=None, query_type: Optional[str]=None, contention_factor: Optional[int]=None, range_opts: Optional[RangeOpts]=None) -> Binary:
        """Encrypt a BSON value with a given key and algorithm.

        Note that exactly one of ``key_id`` or  ``key_alt_name`` must be
        provided.

        :Parameters:
          - `value`: The BSON value to encrypt.
          - `algorithm` (string): The encryption algorithm to use. See
            :class:`Algorithm` for some valid options.
          - `key_id`: Identifies a data key by ``_id`` which must be a
            :class:`~bson.binary.Binary` with subtype 4 (
            :attr:`~bson.binary.UUID_SUBTYPE`).
          - `key_alt_name`: Identifies a key vault document by 'keyAltName'.
          - `query_type` (str): The query type to execute. See :class:`QueryType` for valid options.
          - `contention_factor` (int): The contention factor to use
            when the algorithm is :attr:`Algorithm.INDEXED`.  An integer value
            *must* be given when the :attr:`Algorithm.INDEXED` algorithm is
            used.
          - `range_opts`: Experimental only, not intended for public use.

        :Returns:
          The encrypted value, a :class:`~bson.binary.Binary` with subtype 6.

        .. versionchanged:: 4.2
           Added the `query_type` and `contention_factor` parameters.
        """
        return cast(Binary, self._encrypt_helper(value=value, algorithm=algorithm, key_id=key_id, key_alt_name=key_alt_name, query_type=query_type, contention_factor=contention_factor, range_opts=range_opts, is_expression=False))

    def encrypt_expression(self, expression: Mapping[str, Any], algorithm: str, key_id: Optional[Binary]=None, key_alt_name: Optional[str]=None, query_type: Optional[str]=None, contention_factor: Optional[int]=None, range_opts: Optional[RangeOpts]=None) -> RawBSONDocument:
        """Encrypt a BSON expression with a given key and algorithm.

        Note that exactly one of ``key_id`` or  ``key_alt_name`` must be
        provided.

        :Parameters:
          - `expression`: The BSON aggregate or match expression to encrypt.
          - `algorithm` (string): The encryption algorithm to use. See
            :class:`Algorithm` for some valid options.
          - `key_id`: Identifies a data key by ``_id`` which must be a
            :class:`~bson.binary.Binary` with subtype 4 (
            :attr:`~bson.binary.UUID_SUBTYPE`).
          - `key_alt_name`: Identifies a key vault document by 'keyAltName'.
          - `query_type` (str): The query type to execute. See
            :class:`QueryType` for valid options.
          - `contention_factor` (int): The contention factor to use
            when the algorithm is :attr:`Algorithm.INDEXED`.  An integer value
            *must* be given when the :attr:`Algorithm.INDEXED` algorithm is
            used.
          - `range_opts`: Experimental only, not intended for public use.

        :Returns:
          The encrypted expression, a :class:`~bson.RawBSONDocument`.

        .. versionadded:: 4.4
        """
        return cast(RawBSONDocument, self._encrypt_helper(value=expression, algorithm=algorithm, key_id=key_id, key_alt_name=key_alt_name, query_type=query_type, contention_factor=contention_factor, range_opts=range_opts, is_expression=True))

    def decrypt(self, value: Binary) -> Any:
        """Decrypt an encrypted value.

        :Parameters:
          - `value` (Binary): The encrypted value, a
            :class:`~bson.binary.Binary` with subtype 6.

        :Returns:
          The decrypted BSON value.
        """
        self._check_closed()
        if not (isinstance(value, Binary) and value.subtype == 6):
            raise TypeError('value to decrypt must be a bson.binary.Binary with subtype 6')
        with _wrap_encryption_errors():
            doc = encode({'v': value})
            decrypted_doc = self._encryption.decrypt(doc)
            return decode(decrypted_doc, codec_options=self._codec_options)['v']

    def get_key(self, id: Binary) -> Optional[RawBSONDocument]:
        """Get a data key by id.

        :Parameters:
          - `id` (Binary): The UUID of a key a which must be a
            :class:`~bson.binary.Binary` with subtype 4 (
            :attr:`~bson.binary.UUID_SUBTYPE`).

        :Returns:
          The key document.

        .. versionadded:: 4.2
        """
        self._check_closed()
        assert self._key_vault_coll is not None
        return self._key_vault_coll.find_one({'_id': id})

    def get_keys(self) -> Cursor[RawBSONDocument]:
        """Get all of the data keys.

        :Returns:
          An instance of :class:`~pymongo.cursor.Cursor` over the data key
          documents.

        .. versionadded:: 4.2
        """
        self._check_closed()
        assert self._key_vault_coll is not None
        return self._key_vault_coll.find({})

    def delete_key(self, id: Binary) -> DeleteResult:
        """Delete a key document in the key vault collection that has the given ``key_id``.

        :Parameters:
          - `id` (Binary): The UUID of a key a which must be a
            :class:`~bson.binary.Binary` with subtype 4 (
            :attr:`~bson.binary.UUID_SUBTYPE`).

        :Returns:
          The delete result.

        .. versionadded:: 4.2
        """
        self._check_closed()
        assert self._key_vault_coll is not None
        return self._key_vault_coll.delete_one({'_id': id})

    def add_key_alt_name(self, id: Binary, key_alt_name: str) -> Any:
        """Add ``key_alt_name`` to the set of alternate names in the key document with UUID ``key_id``.

        :Parameters:
          - ``id``: The UUID of a key a which must be a
            :class:`~bson.binary.Binary` with subtype 4 (
            :attr:`~bson.binary.UUID_SUBTYPE`).
          - ``key_alt_name``: The key alternate name to add.

        :Returns:
          The previous version of the key document.

        .. versionadded:: 4.2
        """
        self._check_closed()
        update = {'$addToSet': {'keyAltNames': key_alt_name}}
        assert self._key_vault_coll is not None
        return self._key_vault_coll.find_one_and_update({'_id': id}, update)

    def get_key_by_alt_name(self, key_alt_name: str) -> Optional[RawBSONDocument]:
        """Get a key document in the key vault collection that has the given ``key_alt_name``.

        :Parameters:
          - `key_alt_name`: (str): The key alternate name of the key to get.

        :Returns:
          The key document.

        .. versionadded:: 4.2
        """
        self._check_closed()
        assert self._key_vault_coll is not None
        return self._key_vault_coll.find_one({'keyAltNames': key_alt_name})

    def remove_key_alt_name(self, id: Binary, key_alt_name: str) -> Optional[RawBSONDocument]:
        """Remove ``key_alt_name`` from the set of keyAltNames in the key document with UUID ``id``.

        Also removes the ``keyAltNames`` field from the key document if it would otherwise be empty.

        :Parameters:
          - ``id``: The UUID of a key a which must be a
            :class:`~bson.binary.Binary` with subtype 4 (
            :attr:`~bson.binary.UUID_SUBTYPE`).
          - ``key_alt_name``: The key alternate name to remove.

        :Returns:
          Returns the previous version of the key document.

        .. versionadded:: 4.2
        """
        self._check_closed()
        pipeline = [{'$set': {'keyAltNames': {'$cond': [{'$eq': ['$keyAltNames', [key_alt_name]]}, '$$REMOVE', {'$filter': {'input': '$keyAltNames', 'cond': {'$ne': ['$$this', key_alt_name]}}}]}}}]
        assert self._key_vault_coll is not None
        return self._key_vault_coll.find_one_and_update({'_id': id}, pipeline)

    def rewrap_many_data_key(self, filter: Mapping[str, Any], provider: Optional[str]=None, master_key: Optional[Mapping[str, Any]]=None) -> RewrapManyDataKeyResult:
        """Decrypts and encrypts all matching data keys in the key vault with a possibly new `master_key` value.

        :Parameters:
          - `filter`: A document used to filter the data keys.
          - `provider`: The new KMS provider to use to encrypt the data keys,
            or ``None`` to use the current KMS provider(s).
          - ``master_key``: The master key fields corresponding to the new KMS
            provider when ``provider`` is not ``None``.

        :Returns:
          A :class:`RewrapManyDataKeyResult`.

        This method allows you to re-encrypt all of your data-keys with a new CMK, or master key.
        Note that this does *not* require re-encrypting any of the data in your encrypted collections,
        but rather refreshes the key that protects the keys that encrypt the data:

        .. code-block:: python

           client_encryption.rewrap_many_data_key(
               filter={"keyAltNames": "optional filter for which keys you want to update"},
               master_key={
                   "provider": "azure",  # replace with your cloud provider
                   "master_key": {
                       # put the rest of your master_key options here
                       "key": "<your new key>"
                   },
               },
           )

        .. versionadded:: 4.2
        """
        if master_key is not None and provider is None:
            raise ConfigurationError('A provider must be given if a master_key is given')
        self._check_closed()
        with _wrap_encryption_errors():
            raw_result = self._encryption.rewrap_many_data_key(filter, provider, master_key)
            if raw_result is None:
                return RewrapManyDataKeyResult()
        raw_doc = RawBSONDocument(raw_result, DEFAULT_RAW_BSON_OPTIONS)
        replacements = []
        for key in raw_doc['v']:
            update_model = {'$set': {'keyMaterial': key['keyMaterial'], 'masterKey': key['masterKey']}, '$currentDate': {'updateDate': True}}
            op = UpdateOne({'_id': key['_id']}, update_model)
            replacements.append(op)
        if not replacements:
            return RewrapManyDataKeyResult()
        assert self._key_vault_coll is not None
        result = self._key_vault_coll.bulk_write(replacements)
        return RewrapManyDataKeyResult(result)

    def __enter__(self) -> ClientEncryption[_DocumentType]:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _check_closed(self) -> None:
        if self._encryption is None:
            raise InvalidOperation('Cannot use closed ClientEncryption')

    def close(self) -> None:
        """Release resources.

        Note that using this class in a with-statement will automatically call
        :meth:`close`::

            with ClientEncryption(...) as client_encryption:
                encrypted = client_encryption.encrypt(value, ...)
                decrypted = client_encryption.decrypt(encrypted)

        """
        if self._io_callbacks:
            self._io_callbacks.close()
            self._encryption.close()
            self._io_callbacks = None
            self._encryption = None