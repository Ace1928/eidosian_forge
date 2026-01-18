from __future__ import absolute_import, division, print_function
from ansible_collections.community.routeros.plugins.module_utils.version import LooseVersion
class APIData(object):

    def __init__(self, unversioned=None, versioned=None):
        if (unversioned is None) == (versioned is None):
            raise ValueError('either unversioned or versioned must be provided')
        self.unversioned = unversioned
        self.versioned = versioned
        if self.unversioned is not None:
            self.needs_version = self.unversioned.needs_version
            self.fully_understood = self.unversioned.fully_understood
        else:
            self.needs_version = self.versioned is not None
            self.fully_understood = False
            for dummy, dummy, unversioned in self.versioned:
                if unversioned and (not isinstance(unversioned, str)) and unversioned.fully_understood:
                    self.fully_understood = True
                    break
        self._current = None if self.needs_version else self.unversioned

    def _select(self, data, api_version):
        if data is None:
            self._current = None
            return (False, None)
        if isinstance(data, str):
            self._current = None
            return (False, data)
        self._current = data.specialize_for_version(api_version)
        return (self._current.fully_understood, None)

    def provide_version(self, version):
        if not self.needs_version:
            return (self.unversioned.fully_understood, None)
        api_version = LooseVersion(version)
        if self.unversioned is not None:
            self._current = self.unversioned.specialize_for_version(api_version)
            return (self._current.fully_understood, None)
        for other_version, comparator, data in self.versioned:
            if other_version == '*' and comparator == '*':
                return self._select(data, api_version)
            other_api_version = LooseVersion(other_version)
            if _compare(api_version, other_api_version, comparator):
                return self._select(data, api_version)
        self._current = None
        return (False, None)

    def get_data(self):
        if self._current is None:
            raise ValueError('either provide_version() was not called or it returned False')
        return self._current