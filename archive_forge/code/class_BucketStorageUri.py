import boto
import os
import sys
import textwrap
from boto.s3.deletemarker import DeleteMarker
from boto.exception import BotoClientError
from boto.exception import InvalidUriError
class BucketStorageUri(StorageUri):
    """
    StorageUri subclass that handles bucket storage providers.
    Callers should instantiate this class by calling boto.storage_uri().
    """
    delim = '/'
    capabilities = set([])

    def __init__(self, scheme, bucket_name=None, object_name=None, debug=0, connection_args=None, suppress_consec_slashes=True, version_id=None, generation=None, is_latest=False):
        """Instantiate a BucketStorageUri from scheme,bucket,object tuple.

        @type scheme: string
        @param scheme: URI scheme naming the storage provider (gs, s3, etc.)
        @type bucket_name: string
        @param bucket_name: bucket name
        @type object_name: string
        @param object_name: object name, excluding generation/version.
        @type debug: int
        @param debug: debug level to pass in to connection (range 0..2)
        @type connection_args: map
        @param connection_args: optional map containing args to be
            passed to {S3,GS}Connection constructor (e.g., to override
            https_connection_factory).
        @param suppress_consec_slashes: If provided, controls whether
            consecutive slashes will be suppressed in key paths.
        @param version_id: Object version id (S3-specific).
        @param generation: Object generation number (GCS-specific).
        @param is_latest: boolean indicating that a versioned object is the
            current version

        After instantiation the components are available in the following
        fields: scheme, bucket_name, object_name, version_id, generation,
        is_latest, versionless_uri, version_specific_uri, uri.
        Note: If instantiated without version info, the string representation
        for a URI stays versionless; similarly, if instantiated with version
        info, the string representation for a URI stays version-specific. If you
        call one of the uri.set_contents_from_xyz() methods, a specific object
        version will be created, and its version-specific URI string can be
        retrieved from version_specific_uri even if the URI was instantiated
        without version info.
        """
        self.scheme = scheme
        self.bucket_name = bucket_name
        self.object_name = object_name
        self.debug = debug
        if connection_args:
            self.connection_args = connection_args
        self.suppress_consec_slashes = suppress_consec_slashes
        self.version_id = version_id
        self.generation = generation and int(generation)
        self.is_latest = is_latest
        self.is_version_specific = bool(self.generation) or bool(version_id)
        self._build_uri_strings()

    def _build_uri_strings(self):
        if self.bucket_name and self.object_name:
            self.versionless_uri = '%s://%s/%s' % (self.scheme, self.bucket_name, self.object_name)
            if self.generation:
                self.version_specific_uri = '%s#%s' % (self.versionless_uri, self.generation)
            elif self.version_id:
                self.version_specific_uri = '%s#%s' % (self.versionless_uri, self.version_id)
            if self.is_version_specific:
                self.uri = self.version_specific_uri
            else:
                self.uri = self.versionless_uri
        elif self.bucket_name:
            self.uri = '%s://%s/' % (self.scheme, self.bucket_name)
        else:
            self.uri = '%s://' % self.scheme

    def _update_from_key(self, key):
        self._update_from_values(getattr(key, 'version_id', None), getattr(key, 'generation', None), getattr(key, 'is_latest', None), getattr(key, 'md5', None))

    def _update_from_values(self, version_id, generation, is_latest, md5):
        self.version_id = version_id
        self.generation = generation
        self.is_latest = is_latest
        self._build_uri_strings()
        self.md5 = md5

    def get_key(self, validate=False, headers=None, version_id=None):
        self._check_object_uri('get_key')
        bucket = self.get_bucket(validate, headers)
        if self.get_provider().name == 'aws':
            key = bucket.get_key(self.object_name, headers, version_id=version_id or self.version_id)
        elif self.get_provider().name == 'google':
            key = bucket.get_key(self.object_name, headers, generation=self.generation)
        self.check_response(key, 'key', self.uri)
        return key

    def delete_key(self, validate=False, headers=None, version_id=None, mfa_token=None):
        self._check_object_uri('delete_key')
        bucket = self.get_bucket(validate, headers)
        if self.get_provider().name == 'aws':
            version_id = version_id or self.version_id
            return bucket.delete_key(self.object_name, headers, version_id, mfa_token)
        elif self.get_provider().name == 'google':
            return bucket.delete_key(self.object_name, headers, generation=self.generation)

    def clone_replace_name(self, new_name):
        """Instantiate a BucketStorageUri from the current BucketStorageUri,
        but replacing the object_name.

        @type new_name: string
        @param new_name: new object name
        """
        self._check_bucket_uri('clone_replace_name')
        return BucketStorageUri(self.scheme, bucket_name=self.bucket_name, object_name=new_name, debug=self.debug, suppress_consec_slashes=self.suppress_consec_slashes)

    def clone_replace_key(self, key):
        """Instantiate a BucketStorageUri from the current BucketStorageUri, by
        replacing the object name with the object name and other metadata found
        in the given Key object (including generation).

        @type key: Key
        @param key: key for the new StorageUri to represent
        """
        self._check_bucket_uri('clone_replace_key')
        version_id = None
        generation = None
        is_latest = False
        if hasattr(key, 'version_id'):
            version_id = key.version_id
        if hasattr(key, 'generation'):
            generation = key.generation
        if hasattr(key, 'is_latest'):
            is_latest = key.is_latest
        return BucketStorageUri(key.provider.get_provider_name(), bucket_name=key.bucket.name, object_name=key.name, debug=self.debug, suppress_consec_slashes=self.suppress_consec_slashes, version_id=version_id, generation=generation, is_latest=is_latest)

    def get_acl(self, validate=False, headers=None, version_id=None):
        """returns a bucket's acl"""
        self._check_bucket_uri('get_acl')
        bucket = self.get_bucket(validate, headers)
        key_name = self.object_name or ''
        if self.get_provider().name == 'aws':
            version_id = version_id or self.version_id
            acl = bucket.get_acl(key_name, headers, version_id)
        else:
            acl = bucket.get_acl(key_name, headers, generation=self.generation)
        self.check_response(acl, 'acl', self.uri)
        return acl

    def get_def_acl(self, validate=False, headers=None):
        """returns a bucket's default object acl"""
        self._check_bucket_uri('get_def_acl')
        bucket = self.get_bucket(validate, headers)
        acl = bucket.get_def_acl(headers)
        self.check_response(acl, 'acl', self.uri)
        return acl

    def get_cors(self, validate=False, headers=None):
        """returns a bucket's CORS XML"""
        self._check_bucket_uri('get_cors')
        bucket = self.get_bucket(validate, headers)
        cors = bucket.get_cors(headers)
        self.check_response(cors, 'cors', self.uri)
        return cors

    def set_cors(self, cors, validate=False, headers=None):
        """sets or updates a bucket's CORS XML"""
        self._check_bucket_uri('set_cors ')
        bucket = self.get_bucket(validate, headers)
        if self.scheme == 's3':
            bucket.set_cors(cors, headers)
        else:
            bucket.set_cors(cors.to_xml(), headers)

    def get_location(self, validate=False, headers=None):
        self._check_bucket_uri('get_location')
        bucket = self.get_bucket(validate, headers)
        return bucket.get_location(headers)

    def get_storage_class(self, validate=False, headers=None):
        self._check_bucket_uri('get_storage_class')
        if self.scheme != 'gs':
            raise ValueError('get_storage_class() not supported for %s URIs.' % self.scheme)
        bucket = self.get_bucket(validate, headers)
        return bucket.get_storage_class(headers)

    def set_storage_class(self, storage_class, validate=False, headers=None):
        """Updates a bucket's storage class."""
        self._check_bucket_uri('set_storage_class')
        if self.scheme != 'gs':
            raise ValueError('set_storage_class() not supported for %s URIs.' % self.scheme)
        bucket = self.get_bucket(validate, headers)
        bucket.set_storage_class(storage_class, headers)

    def get_subresource(self, subresource, validate=False, headers=None, version_id=None):
        self._check_bucket_uri('get_subresource')
        bucket = self.get_bucket(validate, headers)
        return bucket.get_subresource(subresource, self.object_name, headers, version_id)

    def add_group_email_grant(self, permission, email_address, recursive=False, validate=False, headers=None):
        self._check_bucket_uri('add_group_email_grant')
        if self.scheme != 'gs':
            raise ValueError('add_group_email_grant() not supported for %s URIs.' % self.scheme)
        if self.object_name:
            if recursive:
                raise ValueError('add_group_email_grant() on key-ful URI cannot specify recursive=True')
            key = self.get_key(validate, headers)
            self.check_response(key, 'key', self.uri)
            key.add_group_email_grant(permission, email_address, headers)
        elif self.bucket_name:
            bucket = self.get_bucket(validate, headers)
            bucket.add_group_email_grant(permission, email_address, recursive, headers)
        else:
            raise InvalidUriError('add_group_email_grant() on bucket-less URI %s' % self.uri)

    def add_email_grant(self, permission, email_address, recursive=False, validate=False, headers=None):
        self._check_bucket_uri('add_email_grant')
        if not self.object_name:
            bucket = self.get_bucket(validate, headers)
            bucket.add_email_grant(permission, email_address, recursive, headers)
        else:
            key = self.get_key(validate, headers)
            self.check_response(key, 'key', self.uri)
            key.add_email_grant(permission, email_address)

    def add_user_grant(self, permission, user_id, recursive=False, validate=False, headers=None):
        self._check_bucket_uri('add_user_grant')
        if not self.object_name:
            bucket = self.get_bucket(validate, headers)
            bucket.add_user_grant(permission, user_id, recursive, headers)
        else:
            key = self.get_key(validate, headers)
            self.check_response(key, 'key', self.uri)
            key.add_user_grant(permission, user_id)

    def list_grants(self, headers=None):
        self._check_bucket_uri('list_grants ')
        bucket = self.get_bucket(headers)
        return bucket.list_grants(headers)

    def is_file_uri(self):
        """Returns True if this URI names a file or directory."""
        return False

    def is_cloud_uri(self):
        """Returns True if this URI names a bucket or object."""
        return True

    def names_container(self):
        """
        Returns True if this URI names a directory or bucket. Will return
        False for bucket subdirs; providing bucket subdir semantics needs to
        be done by the caller (like gsutil does).
        """
        return bool(not self.object_name)

    def names_singleton(self):
        """Returns True if this URI names a file or object."""
        return bool(self.object_name)

    def names_directory(self):
        """Returns True if this URI names a directory."""
        return False

    def names_provider(self):
        """Returns True if this URI names a provider."""
        return bool(not self.bucket_name)

    def names_bucket(self):
        """Returns True if this URI names a bucket."""
        return bool(self.bucket_name) and bool(not self.object_name)

    def names_file(self):
        """Returns True if this URI names a file."""
        return False

    def names_object(self):
        """Returns True if this URI names an object."""
        return self.names_singleton()

    def is_stream(self):
        """Returns True if this URI represents input/output stream."""
        return False

    def create_bucket(self, headers=None, location='', policy=None, storage_class=None):
        self._check_bucket_uri('create_bucket ')
        conn = self.connect()
        if self.scheme == 'gs':
            return conn.create_bucket(self.bucket_name, headers, location, policy, storage_class)
        else:
            return conn.create_bucket(self.bucket_name, headers, location, policy)

    def delete_bucket(self, headers=None):
        self._check_bucket_uri('delete_bucket')
        conn = self.connect()
        return conn.delete_bucket(self.bucket_name, headers)

    def get_all_buckets(self, headers=None):
        conn = self.connect()
        return conn.get_all_buckets(headers)

    def get_provider(self):
        conn = self.connect()
        provider = conn.provider
        self.check_response(provider, 'provider', self.uri)
        return provider

    def set_acl(self, acl_or_str, key_name='', validate=False, headers=None, version_id=None, if_generation=None, if_metageneration=None):
        """Sets or updates a bucket's ACL."""
        self._check_bucket_uri('set_acl')
        key_name = key_name or self.object_name or ''
        bucket = self.get_bucket(validate, headers)
        if self.generation:
            bucket.set_acl(acl_or_str, key_name, headers, generation=self.generation, if_generation=if_generation, if_metageneration=if_metageneration)
        else:
            version_id = version_id or self.version_id
            bucket.set_acl(acl_or_str, key_name, headers, version_id)

    def set_xml_acl(self, xmlstring, key_name='', validate=False, headers=None, version_id=None, if_generation=None, if_metageneration=None):
        """Sets or updates a bucket's ACL with an XML string."""
        self._check_bucket_uri('set_xml_acl')
        key_name = key_name or self.object_name or ''
        bucket = self.get_bucket(validate, headers)
        if self.generation:
            bucket.set_xml_acl(xmlstring, key_name, headers, generation=self.generation, if_generation=if_generation, if_metageneration=if_metageneration)
        else:
            version_id = version_id or self.version_id
            bucket.set_xml_acl(xmlstring, key_name, headers, version_id=version_id)

    def set_def_xml_acl(self, xmlstring, validate=False, headers=None):
        """Sets or updates a bucket's default object ACL with an XML string."""
        self._check_bucket_uri('set_def_xml_acl')
        self.get_bucket(validate, headers).set_def_xml_acl(xmlstring, headers)

    def set_def_acl(self, acl_or_str, validate=False, headers=None, version_id=None):
        """Sets or updates a bucket's default object ACL."""
        self._check_bucket_uri('set_def_acl')
        self.get_bucket(validate, headers).set_def_acl(acl_or_str, headers)

    def set_canned_acl(self, acl_str, validate=False, headers=None, version_id=None):
        """Sets or updates a bucket's acl to a predefined (canned) value."""
        self._check_object_uri('set_canned_acl')
        self._warn_about_args('set_canned_acl', version_id=version_id)
        key = self.get_key(validate, headers)
        self.check_response(key, 'key', self.uri)
        key.set_canned_acl(acl_str, headers)

    def set_def_canned_acl(self, acl_str, validate=False, headers=None, version_id=None):
        """Sets or updates a bucket's default object acl to a predefined
           (canned) value."""
        self._check_bucket_uri('set_def_canned_acl ')
        key = self.get_key(validate, headers)
        self.check_response(key, 'key', self.uri)
        key.set_def_canned_acl(acl_str, headers, version_id)

    def set_subresource(self, subresource, value, validate=False, headers=None, version_id=None):
        self._check_bucket_uri('set_subresource')
        bucket = self.get_bucket(validate, headers)
        bucket.set_subresource(subresource, value, self.object_name, headers, version_id)

    def set_contents_from_string(self, s, headers=None, replace=True, cb=None, num_cb=10, policy=None, md5=None, reduced_redundancy=False):
        self._check_object_uri('set_contents_from_string')
        key = self.new_key(headers=headers)
        if self.scheme == 'gs':
            if reduced_redundancy:
                sys.stderr.write('Warning: GCS does not support reduced_redundancy; argument ignored by set_contents_from_string')
            result = key.set_contents_from_string(s, headers, replace, cb, num_cb, policy, md5)
        else:
            result = key.set_contents_from_string(s, headers, replace, cb, num_cb, policy, md5, reduced_redundancy)
        self._update_from_key(key)
        return result

    def set_contents_from_file(self, fp, headers=None, replace=True, cb=None, num_cb=10, policy=None, md5=None, size=None, rewind=False, res_upload_handler=None):
        self._check_object_uri('set_contents_from_file')
        key = self.new_key(headers=headers)
        if self.scheme == 'gs':
            result = key.set_contents_from_file(fp, headers, replace, cb, num_cb, policy, md5, size=size, rewind=rewind, res_upload_handler=res_upload_handler)
            if res_upload_handler:
                self._update_from_values(None, res_upload_handler.generation, None, md5)
        else:
            self._warn_about_args('set_contents_from_file', res_upload_handler=res_upload_handler)
            result = key.set_contents_from_file(fp, headers, replace, cb, num_cb, policy, md5, size=size, rewind=rewind)
        self._update_from_key(key)
        return result

    def set_contents_from_stream(self, fp, headers=None, replace=True, cb=None, policy=None, reduced_redundancy=False):
        self._check_object_uri('set_contents_from_stream')
        dst_key = self.new_key(False, headers)
        result = dst_key.set_contents_from_stream(fp, headers, replace, cb, policy=policy, reduced_redundancy=reduced_redundancy)
        self._update_from_key(dst_key)
        return result

    def copy_key(self, src_bucket_name, src_key_name, metadata=None, src_version_id=None, storage_class='STANDARD', preserve_acl=False, encrypt_key=False, headers=None, query_args=None, src_generation=None):
        """Returns newly created key."""
        self._check_object_uri('copy_key')
        dst_bucket = self.get_bucket(validate=False, headers=headers)
        if src_generation:
            return dst_bucket.copy_key(new_key_name=self.object_name, src_bucket_name=src_bucket_name, src_key_name=src_key_name, metadata=metadata, storage_class=storage_class, preserve_acl=preserve_acl, encrypt_key=encrypt_key, headers=headers, query_args=query_args, src_generation=src_generation)
        else:
            return dst_bucket.copy_key(new_key_name=self.object_name, src_bucket_name=src_bucket_name, src_key_name=src_key_name, metadata=metadata, src_version_id=src_version_id, storage_class=storage_class, preserve_acl=preserve_acl, encrypt_key=encrypt_key, headers=headers, query_args=query_args)

    def enable_logging(self, target_bucket, target_prefix=None, validate=False, headers=None, version_id=None):
        self._check_bucket_uri('enable_logging')
        bucket = self.get_bucket(validate, headers)
        bucket.enable_logging(target_bucket, target_prefix, headers=headers)

    def disable_logging(self, validate=False, headers=None, version_id=None):
        self._check_bucket_uri('disable_logging')
        bucket = self.get_bucket(validate, headers)
        bucket.disable_logging(headers=headers)

    def get_logging_config(self, validate=False, headers=None, version_id=None):
        self._check_bucket_uri('get_logging_config')
        bucket = self.get_bucket(validate, headers)
        return bucket.get_logging_config(headers=headers)

    def set_website_config(self, main_page_suffix=None, error_key=None, validate=False, headers=None):
        self._check_bucket_uri('set_website_config')
        bucket = self.get_bucket(validate, headers)
        if not (main_page_suffix or error_key):
            bucket.delete_website_configuration(headers)
        else:
            bucket.configure_website(main_page_suffix, error_key, headers)

    def get_website_config(self, validate=False, headers=None):
        self._check_bucket_uri('get_website_config')
        bucket = self.get_bucket(validate, headers)
        return bucket.get_website_configuration(headers)

    def get_versioning_config(self, headers=None):
        self._check_bucket_uri('get_versioning_config')
        bucket = self.get_bucket(False, headers)
        return bucket.get_versioning_status(headers)

    def configure_versioning(self, enabled, headers=None):
        self._check_bucket_uri('configure_versioning')
        bucket = self.get_bucket(False, headers)
        return bucket.configure_versioning(enabled, headers)

    def set_metadata(self, metadata_plus, metadata_minus, preserve_acl, headers=None):
        return self.get_key(False).set_remote_metadata(metadata_plus, metadata_minus, preserve_acl, headers=headers)

    def compose(self, components, content_type=None, headers=None):
        self._check_object_uri('compose')
        component_keys = []
        for suri in components:
            component_keys.append(suri.new_key())
            component_keys[-1].generation = suri.generation
        self.generation = self.new_key().compose(component_keys, content_type=content_type, headers=headers)
        self._build_uri_strings()
        return self

    def get_lifecycle_config(self, validate=False, headers=None):
        """Returns a bucket's lifecycle configuration."""
        self._check_bucket_uri('get_lifecycle_config')
        bucket = self.get_bucket(validate, headers)
        lifecycle_config = bucket.get_lifecycle_config(headers)
        self.check_response(lifecycle_config, 'lifecycle', self.uri)
        return lifecycle_config

    def configure_lifecycle(self, lifecycle_config, validate=False, headers=None):
        """Sets or updates a bucket's lifecycle configuration."""
        self._check_bucket_uri('configure_lifecycle')
        bucket = self.get_bucket(validate, headers)
        bucket.configure_lifecycle(lifecycle_config, headers)

    def get_billing_config(self, headers=None):
        self._check_bucket_uri('get_billing_config')
        if self.scheme != 'gs':
            raise ValueError('get_billing_config() not supported for %s URIs.' % self.scheme)
        bucket = self.get_bucket(False, headers)
        return bucket.get_billing_config(headers)

    def configure_billing(self, requester_pays=False, validate=False, headers=None):
        """Sets or updates a bucket's billing configuration."""
        self._check_bucket_uri('configure_billing')
        if self.scheme != 'gs':
            raise ValueError('configure_billing() not supported for %s URIs.' % self.scheme)
        bucket = self.get_bucket(validate, headers)
        bucket.configure_billing(requester_pays=requester_pays, headers=headers)

    def get_encryption_config(self, validate=False, headers=None):
        """Returns a GCS bucket's encryption configuration."""
        self._check_bucket_uri('get_encryption_config')
        if self.scheme != 'gs':
            raise ValueError('get_encryption_config() not supported for %s URIs.' % self.scheme)
        bucket = self.get_bucket(validate, headers)
        return bucket.get_encryption_config(headers=headers)

    def set_encryption_config(self, default_kms_key_name=None, validate=False, headers=None):
        """Sets a GCS bucket's encryption configuration."""
        self._check_bucket_uri('set_encryption_config')
        bucket = self.get_bucket(validate, headers)
        bucket.set_encryption_config(default_kms_key_name=default_kms_key_name, headers=headers)

    def exists(self, headers=None):
        """Returns True if the object exists or False if it doesn't"""
        if not self.object_name:
            raise InvalidUriError('exists on object-less URI (%s)' % self.uri)
        bucket = self.get_bucket(headers)
        key = bucket.get_key(self.object_name, headers=headers)
        return bool(key)