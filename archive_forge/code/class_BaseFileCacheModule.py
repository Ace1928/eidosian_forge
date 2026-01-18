from __future__ import (absolute_import, division, print_function)
import copy
import errno
import os
import tempfile
import time
from abc import abstractmethod
from collections.abc import MutableMapping
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.plugins import AnsiblePlugin
from ansible.plugins.loader import cache_loader
from ansible.utils.collection_loader import resource_from_fqcr
from ansible.utils.display import Display
class BaseFileCacheModule(BaseCacheModule):
    """
    A caching module backed by file based storage.
    """

    def __init__(self, *args, **kwargs):
        try:
            super(BaseFileCacheModule, self).__init__(*args, **kwargs)
            self._cache_dir = self._get_cache_connection(self.get_option('_uri'))
            self._timeout = float(self.get_option('_timeout'))
        except KeyError:
            self._cache_dir = self._get_cache_connection(C.CACHE_PLUGIN_CONNECTION)
            self._timeout = float(C.CACHE_PLUGIN_TIMEOUT)
        self.plugin_name = resource_from_fqcr(self.__module__)
        self._cache = {}
        self.validate_cache_connection()

    def _get_cache_connection(self, source):
        if source:
            try:
                return os.path.expanduser(os.path.expandvars(source))
            except TypeError:
                pass

    def validate_cache_connection(self):
        if not self._cache_dir:
            raise AnsibleError("error, '%s' cache plugin requires the 'fact_caching_connection' config option to be set (to a writeable directory path)" % self.plugin_name)
        if not os.path.exists(self._cache_dir):
            try:
                os.makedirs(self._cache_dir)
            except (OSError, IOError) as e:
                raise AnsibleError("error in '%s' cache plugin while trying to create cache dir %s : %s" % (self.plugin_name, self._cache_dir, to_bytes(e)))
        else:
            for x in (os.R_OK, os.W_OK, os.X_OK):
                if not os.access(self._cache_dir, x):
                    raise AnsibleError("error in '%s' cache, configured path (%s) does not have necessary permissions (rwx), disabling plugin" % (self.plugin_name, self._cache_dir))

    def _get_cache_file_name(self, key):
        prefix = self.get_option('_prefix')
        if prefix:
            cachefile = '%s/%s%s' % (self._cache_dir, prefix, key)
        else:
            cachefile = '%s/%s' % (self._cache_dir, key)
        return cachefile

    def get(self, key):
        """ This checks the in memory cache first as the fact was not expired at 'gather time'
        and it would be problematic if the key did expire after some long running tasks and
        user gets 'undefined' error in the same play """
        if key not in self._cache:
            if self.has_expired(key) or key == '':
                raise KeyError
            cachefile = self._get_cache_file_name(key)
            try:
                value = self._load(cachefile)
                self._cache[key] = value
            except ValueError as e:
                display.warning("error in '%s' cache plugin while trying to read %s : %s. Most likely a corrupt file, so erasing and failing." % (self.plugin_name, cachefile, to_bytes(e)))
                self.delete(key)
                raise AnsibleError('The cache file %s was corrupt, or did not otherwise contain valid data. It has been removed, so you can re-run your command now.' % cachefile)
            except (OSError, IOError) as e:
                display.warning("error in '%s' cache plugin while trying to read %s : %s" % (self.plugin_name, cachefile, to_bytes(e)))
                raise KeyError
            except Exception as e:
                raise AnsibleError('Error while decoding the cache file %s: %s' % (cachefile, to_bytes(e)))
        return self._cache.get(key)

    def set(self, key, value):
        self._cache[key] = value
        cachefile = self._get_cache_file_name(key)
        tmpfile_handle, tmpfile_path = tempfile.mkstemp(dir=os.path.dirname(cachefile))
        try:
            try:
                self._dump(value, tmpfile_path)
            except (OSError, IOError) as e:
                display.warning("error in '%s' cache plugin while trying to write to '%s' : %s" % (self.plugin_name, tmpfile_path, to_bytes(e)))
            try:
                os.rename(tmpfile_path, cachefile)
                os.chmod(cachefile, mode=420)
            except (OSError, IOError) as e:
                display.warning("error in '%s' cache plugin while trying to move '%s' to '%s' : %s" % (self.plugin_name, tmpfile_path, cachefile, to_bytes(e)))
        finally:
            try:
                os.unlink(tmpfile_path)
            except OSError:
                pass

    def has_expired(self, key):
        if self._timeout == 0:
            return False
        cachefile = self._get_cache_file_name(key)
        try:
            st = os.stat(cachefile)
        except (OSError, IOError) as e:
            if e.errno == errno.ENOENT:
                return False
            else:
                display.warning("error in '%s' cache plugin while trying to stat %s : %s" % (self.plugin_name, cachefile, to_bytes(e)))
                return False
        if time.time() - st.st_mtime <= self._timeout:
            return False
        if key in self._cache:
            del self._cache[key]
        return True

    def keys(self):
        prefix = self.get_option('_prefix')
        prefix_length = len(prefix)
        keys = []
        for k in os.listdir(self._cache_dir):
            if k.startswith('.') or not k.startswith(prefix):
                continue
            k = k[prefix_length:]
            if not self.has_expired(k):
                keys.append(k)
        return keys

    def contains(self, key):
        cachefile = self._get_cache_file_name(key)
        if key in self._cache:
            return True
        if self.has_expired(key):
            return False
        try:
            os.stat(cachefile)
            return True
        except (OSError, IOError) as e:
            if e.errno == errno.ENOENT:
                return False
            else:
                display.warning("error in '%s' cache plugin while trying to stat %s : %s" % (self.plugin_name, cachefile, to_bytes(e)))

    def delete(self, key):
        try:
            del self._cache[key]
        except KeyError:
            pass
        try:
            os.remove(self._get_cache_file_name(key))
        except (OSError, IOError):
            pass

    def flush(self):
        self._cache = {}
        for key in self.keys():
            self.delete(key)

    def copy(self):
        ret = dict()
        for key in self.keys():
            ret[key] = self.get(key)
        return ret

    @abstractmethod
    def _load(self, filepath):
        """
        Read data from a filepath and return it as a value

        :arg filepath: The filepath to read from.
        :returns: The value stored in the filepath

        This method reads from the file on disk and takes care of any parsing
        and transformation of the data before returning it.  The value
        returned should be what Ansible would expect if it were uncached data.

        .. note:: Filehandles have advantages but calling code doesn't know
            whether this file is text or binary, should be decoded, or accessed via
            a library function.  Therefore the API uses a filepath and opens
            the file inside of the method.
        """
        pass

    @abstractmethod
    def _dump(self, value, filepath):
        """
        Write data to a filepath

        :arg value: The value to store
        :arg filepath: The filepath to store it at
        """
        pass