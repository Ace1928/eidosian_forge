import copy
import boto
import base64
import re
import six
from hashlib import md5
from boto.utils import compute_md5
from boto.utils import find_matching_headers
from boto.utils import merge_headers_by_name
from boto.utils import write_to_fd
from boto.s3.prefix import Prefix
from boto.compat import six
class MockBucketStorageUri(object):
    delim = '/'

    def __init__(self, scheme, bucket_name=None, object_name=None, debug=NOT_IMPL, suppress_consec_slashes=NOT_IMPL, version_id=None, generation=None, is_latest=False):
        self.scheme = scheme
        self.bucket_name = bucket_name
        self.object_name = object_name
        self.suppress_consec_slashes = suppress_consec_slashes
        if self.bucket_name and self.object_name:
            self.uri = '%s://%s/%s' % (self.scheme, self.bucket_name, self.object_name)
        elif self.bucket_name:
            self.uri = '%s://%s/' % (self.scheme, self.bucket_name)
        else:
            self.uri = '%s://' % self.scheme
        self.version_id = version_id
        self.generation = generation and int(generation)
        self.is_version_specific = bool(self.generation) or bool(self.version_id)
        self.is_latest = is_latest
        if bucket_name and object_name:
            self.versionless_uri = '%s://%s/%s' % (scheme, bucket_name, object_name)

    def __repr__(self):
        """Returns string representation of URI."""
        return self.uri

    def acl_class(self):
        return MockAcl

    def canned_acls(self):
        return boto.provider.Provider('aws').canned_acls

    def clone_replace_name(self, new_name):
        return self.__class__(self.scheme, self.bucket_name, new_name)

    def clone_replace_key(self, key):
        return self.__class__(key.provider.get_provider_name(), bucket_name=key.bucket.name, object_name=key.name, suppress_consec_slashes=self.suppress_consec_slashes, version_id=getattr(key, 'version_id', None), generation=getattr(key, 'generation', None), is_latest=getattr(key, 'is_latest', None))

    def connect(self, access_key_id=NOT_IMPL, secret_access_key=NOT_IMPL):
        return mock_connection

    def create_bucket(self, headers=NOT_IMPL, location=NOT_IMPL, policy=NOT_IMPL, storage_class=NOT_IMPL):
        return self.connect().create_bucket(self.bucket_name)

    def delete_bucket(self, headers=NOT_IMPL):
        return self.connect().delete_bucket(self.bucket_name)

    def get_versioning_config(self, headers=NOT_IMPL):
        self.get_bucket().get_versioning_status(headers)

    def has_version(self):
        return issubclass(type(self), MockBucketStorageUri) and (self.version_id is not None or self.generation is not None)

    def delete_key(self, validate=NOT_IMPL, headers=NOT_IMPL, version_id=NOT_IMPL, mfa_token=NOT_IMPL):
        self.get_bucket().delete_key(self.object_name)

    def disable_logging(self, validate=NOT_IMPL, headers=NOT_IMPL, version_id=NOT_IMPL):
        self.get_bucket().disable_logging()

    def enable_logging(self, target_bucket, target_prefix, validate=NOT_IMPL, headers=NOT_IMPL, version_id=NOT_IMPL):
        self.get_bucket().enable_logging(target_bucket)

    def get_logging_config(self, validate=NOT_IMPL, headers=NOT_IMPL, version_id=NOT_IMPL):
        return self.get_bucket().get_logging_config()

    def equals(self, uri):
        return self.uri == uri.uri

    def get_acl(self, validate=NOT_IMPL, headers=NOT_IMPL, version_id=NOT_IMPL):
        return self.get_bucket().get_acl(self.object_name)

    def get_def_acl(self, validate=NOT_IMPL, headers=NOT_IMPL, version_id=NOT_IMPL):
        return self.get_bucket().get_def_acl(self.object_name)

    def get_subresource(self, subresource, validate=NOT_IMPL, headers=NOT_IMPL, version_id=NOT_IMPL):
        return self.get_bucket().get_subresource(subresource, self.object_name)

    def get_all_buckets(self, headers=NOT_IMPL):
        return self.connect().get_all_buckets()

    def get_all_keys(self, validate=NOT_IMPL, headers=NOT_IMPL):
        return self.get_bucket().get_all_keys(self)

    def list_bucket(self, prefix='', delimiter='', headers=NOT_IMPL, all_versions=NOT_IMPL):
        return self.get_bucket().list(prefix=prefix, delimiter=delimiter)

    def get_bucket(self, validate=NOT_IMPL, headers=NOT_IMPL):
        return self.connect().get_bucket(self.bucket_name)

    def get_key(self, validate=NOT_IMPL, headers=NOT_IMPL, version_id=NOT_IMPL):
        return self.get_bucket().get_key(self.object_name)

    def is_file_uri(self):
        return False

    def is_cloud_uri(self):
        return True

    def names_container(self):
        return bool(not self.object_name)

    def names_singleton(self):
        return bool(self.object_name)

    def names_directory(self):
        return False

    def names_provider(self):
        return bool(not self.bucket_name)

    def names_bucket(self):
        return self.names_container()

    def names_file(self):
        return False

    def names_object(self):
        return not self.names_container()

    def is_stream(self):
        return False

    def new_key(self, validate=NOT_IMPL, headers=NOT_IMPL):
        bucket = self.get_bucket()
        return bucket.new_key(self.object_name)

    def set_acl(self, acl_or_str, key_name='', validate=NOT_IMPL, headers=NOT_IMPL, version_id=NOT_IMPL):
        self.get_bucket().set_acl(acl_or_str, key_name)

    def set_def_acl(self, acl_or_str, key_name=NOT_IMPL, validate=NOT_IMPL, headers=NOT_IMPL, version_id=NOT_IMPL):
        self.get_bucket().set_def_acl(acl_or_str)

    def set_subresource(self, subresource, value, validate=NOT_IMPL, headers=NOT_IMPL, version_id=NOT_IMPL):
        self.get_bucket().set_subresource(subresource, value, self.object_name)

    def copy_key(self, src_bucket_name, src_key_name, metadata=NOT_IMPL, src_version_id=NOT_IMPL, storage_class=NOT_IMPL, preserve_acl=NOT_IMPL, encrypt_key=NOT_IMPL, headers=NOT_IMPL, query_args=NOT_IMPL, src_generation=NOT_IMPL):
        dst_bucket = self.get_bucket()
        return dst_bucket.copy_key(new_key_name=self.object_name, src_bucket_name=src_bucket_name, src_key_name=src_key_name)

    def set_contents_from_string(self, s, headers=NOT_IMPL, replace=NOT_IMPL, cb=NOT_IMPL, num_cb=NOT_IMPL, policy=NOT_IMPL, md5=NOT_IMPL, reduced_redundancy=NOT_IMPL):
        key = self.new_key()
        key.set_contents_from_string(s)

    def set_contents_from_file(self, fp, headers=None, replace=NOT_IMPL, cb=NOT_IMPL, num_cb=NOT_IMPL, policy=NOT_IMPL, md5=NOT_IMPL, size=NOT_IMPL, rewind=NOT_IMPL, res_upload_handler=NOT_IMPL):
        key = self.new_key()
        return key.set_contents_from_file(fp, headers=headers)

    def set_contents_from_stream(self, fp, headers=NOT_IMPL, replace=NOT_IMPL, cb=NOT_IMPL, num_cb=NOT_IMPL, policy=NOT_IMPL, reduced_redundancy=NOT_IMPL, query_args=NOT_IMPL, size=NOT_IMPL):
        dst_key.set_contents_from_stream(fp)

    def get_contents_to_file(self, fp, headers=NOT_IMPL, cb=NOT_IMPL, num_cb=NOT_IMPL, torrent=NOT_IMPL, version_id=NOT_IMPL, res_download_handler=NOT_IMPL, response_headers=NOT_IMPL):
        key = self.get_key()
        key.get_contents_to_file(fp)

    def get_contents_to_stream(self, fp, headers=NOT_IMPL, cb=NOT_IMPL, num_cb=NOT_IMPL, version_id=NOT_IMPL):
        key = self.get_key()
        return key.get_contents_to_file(fp)