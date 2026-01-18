import boto
import os
import sys
import textwrap
from boto.s3.deletemarker import DeleteMarker
from boto.exception import BotoClientError
from boto.exception import InvalidUriError
class StorageUri(object):
    """
    Base class for representing storage provider-independent bucket and
    object name with a shorthand URI-like syntax.

    This is an abstract class: the constructor cannot be called (throws an
    exception if you try).
    """
    connection = None
    connection_args = None
    provider_pool = {}

    def __init__(self):
        """Uncallable constructor on abstract base StorageUri class.
        """
        raise BotoClientError('Attempt to instantiate abstract StorageUri class')

    def __repr__(self):
        """Returns string representation of URI."""
        return self.uri

    def equals(self, uri):
        """Returns true if two URIs are equal."""
        return self.uri == uri.uri

    def check_response(self, resp, level, uri):
        if resp is None:
            raise InvalidUriError('\n'.join(textwrap.wrap('Attempt to get %s for "%s" failed. This can happen if the URI refers to a non-existent object or if you meant to operate on a directory (e.g., leaving off -R option on gsutil cp, mv, or ls of a bucket)' % (level, uri), 80)))

    def _check_bucket_uri(self, function_name):
        if issubclass(type(self), BucketStorageUri) and (not self.bucket_name):
            raise InvalidUriError('%s on bucket-less URI (%s)' % (function_name, self.uri))

    def _check_object_uri(self, function_name):
        if issubclass(type(self), BucketStorageUri) and (not self.object_name):
            raise InvalidUriError('%s on object-less URI (%s)' % (function_name, self.uri))

    def _warn_about_args(self, function_name, **args):
        for arg in args:
            if args[arg]:
                sys.stderr.write('Warning: %s ignores argument: %s=%s\n' % (function_name, arg, str(args[arg])))

    def connect(self, access_key_id=None, secret_access_key=None, **kwargs):
        """
        Opens a connection to appropriate provider, depending on provider
        portion of URI. Requires Credentials defined in boto config file (see
        boto/pyami/config.py).
        @type storage_uri: StorageUri
        @param storage_uri: StorageUri specifying a bucket or a bucket+object
        @rtype: L{AWSAuthConnection<boto.gs.connection.AWSAuthConnection>}
        @return: A connection to storage service provider of the given URI.
        """
        connection_args = dict(self.connection_args or ())
        if hasattr(self, 'suppress_consec_slashes') and 'suppress_consec_slashes' not in connection_args:
            connection_args['suppress_consec_slashes'] = self.suppress_consec_slashes
        connection_args.update(kwargs)
        if not self.connection:
            if self.scheme in self.provider_pool:
                self.connection = self.provider_pool[self.scheme]
            elif self.scheme == 's3':
                from boto.s3.connection import S3Connection
                self.connection = S3Connection(access_key_id, secret_access_key, **connection_args)
                self.provider_pool[self.scheme] = self.connection
            elif self.scheme == 'gs':
                from boto.gs.connection import GSConnection
                self.connection = GSConnection(access_key_id, secret_access_key, **connection_args)
                self.provider_pool[self.scheme] = self.connection
            elif self.scheme == 'file':
                from boto.file.connection import FileConnection
                self.connection = FileConnection(self)
            else:
                raise InvalidUriError('Unrecognized scheme "%s"' % self.scheme)
        self.connection.debug = self.debug
        return self.connection

    def has_version(self):
        return issubclass(type(self), BucketStorageUri) and (self.version_id is not None or self.generation is not None)

    def delete_key(self, validate=False, headers=None, version_id=None, mfa_token=None):
        self._check_object_uri('delete_key')
        bucket = self.get_bucket(validate, headers)
        return bucket.delete_key(self.object_name, headers, version_id, mfa_token)

    def list_bucket(self, prefix='', delimiter='', headers=None, all_versions=False):
        self._check_bucket_uri('list_bucket')
        bucket = self.get_bucket(headers=headers)
        if all_versions:
            return (v for v in bucket.list_versions(prefix=prefix, delimiter=delimiter, headers=headers) if not isinstance(v, DeleteMarker))
        else:
            return bucket.list(prefix=prefix, delimiter=delimiter, headers=headers)

    def get_all_keys(self, validate=False, headers=None, prefix=None):
        bucket = self.get_bucket(validate, headers)
        return bucket.get_all_keys(headers)

    def get_bucket(self, validate=False, headers=None):
        self._check_bucket_uri('get_bucket')
        conn = self.connect()
        bucket = conn.get_bucket(self.bucket_name, validate, headers)
        self.check_response(bucket, 'bucket', self.uri)
        return bucket

    def get_key(self, validate=False, headers=None, version_id=None):
        self._check_object_uri('get_key')
        bucket = self.get_bucket(validate, headers)
        key = bucket.get_key(self.object_name, headers, version_id)
        self.check_response(key, 'key', self.uri)
        return key

    def new_key(self, validate=False, headers=None):
        self._check_object_uri('new_key')
        bucket = self.get_bucket(validate, headers)
        return bucket.new_key(self.object_name)

    def get_contents_to_stream(self, fp, headers=None, version_id=None):
        self._check_object_uri('get_key')
        self._warn_about_args('get_key', validate=False)
        key = self.get_key(None, headers)
        self.check_response(key, 'key', self.uri)
        return key.get_contents_to_file(fp, headers, version_id=version_id)

    def get_contents_to_file(self, fp, headers=None, cb=None, num_cb=10, torrent=False, version_id=None, res_download_handler=None, response_headers=None, hash_algs=None):
        self._check_object_uri('get_contents_to_file')
        key = self.get_key(None, headers)
        self.check_response(key, 'key', self.uri)
        if hash_algs:
            key.get_contents_to_file(fp, headers, cb, num_cb, torrent, version_id, res_download_handler, response_headers, hash_algs=hash_algs)
        else:
            key.get_contents_to_file(fp, headers, cb, num_cb, torrent, version_id, res_download_handler, response_headers)

    def get_contents_as_string(self, validate=False, headers=None, cb=None, num_cb=10, torrent=False, version_id=None):
        self._check_object_uri('get_contents_as_string')
        key = self.get_key(validate, headers)
        self.check_response(key, 'key', self.uri)
        return key.get_contents_as_string(headers, cb, num_cb, torrent, version_id)

    def acl_class(self):
        conn = self.connect()
        acl_class = conn.provider.acl_class
        self.check_response(acl_class, 'acl_class', self.uri)
        return acl_class

    def canned_acls(self):
        conn = self.connect()
        canned_acls = conn.provider.canned_acls
        self.check_response(canned_acls, 'canned_acls', self.uri)
        return canned_acls