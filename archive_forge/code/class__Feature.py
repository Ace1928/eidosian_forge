import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
class _Feature:
    """A helper class for `CCompilerOpt` that managing CPU features.

    Attributes
    ----------
    feature_supported : dict
        Dictionary containing all CPU features that supported
        by the platform, according to the specified values in attribute
        `_Config.conf_features` and `_Config.conf_features_partial()`

    feature_min : set
        The minimum support of CPU features, according to
        the specified values in attribute `_Config.conf_min_features`.
    """

    def __init__(self):
        if hasattr(self, 'feature_is_cached'):
            return
        self.feature_supported = pfeatures = self.conf_features_partial()
        for feature_name in list(pfeatures.keys()):
            feature = pfeatures[feature_name]
            cfeature = self.conf_features[feature_name]
            feature.update({k: v for k, v in cfeature.items() if k not in feature})
            disabled = feature.get('disable')
            if disabled is not None:
                pfeatures.pop(feature_name)
                self.dist_log("feature '%s' is disabled," % feature_name, disabled, stderr=True)
                continue
            for option in ('implies', 'group', 'detect', 'headers', 'flags', 'extra_checks'):
                oval = feature.get(option)
                if isinstance(oval, str):
                    feature[option] = oval.split()
        self.feature_min = set()
        min_f = self.conf_min_features.get(self.cc_march, '')
        for F in min_f.upper().split():
            if F in self.feature_supported:
                self.feature_min.add(F)
        self.feature_is_cached = True

    def feature_names(self, names=None, force_flags=None, macros=[]):
        """
        Returns a set of CPU feature names that supported by platform and the **C** compiler.

        Parameters
        ----------
        names : sequence or None, optional
            Specify certain CPU features to test it against the **C** compiler.
            if None(default), it will test all current supported features.
            **Note**: feature names must be in upper-case.

        force_flags : list or None, optional
            If None(default), default compiler flags for every CPU feature will
            be used during the test.

        macros : list of tuples, optional
            A list of C macro definitions.
        """
        assert names is None or (not isinstance(names, str) and hasattr(names, '__iter__'))
        assert force_flags is None or isinstance(force_flags, list)
        if names is None:
            names = self.feature_supported.keys()
        supported_names = set()
        for f in names:
            if self.feature_is_supported(f, force_flags=force_flags, macros=macros):
                supported_names.add(f)
        return supported_names

    def feature_is_exist(self, name):
        """
        Returns True if a certain feature is exist and covered within
        ``_Config.conf_features``.

        Parameters
        ----------
        'name': str
            feature name in uppercase.
        """
        assert name.isupper()
        return name in self.conf_features

    def feature_sorted(self, names, reverse=False):
        """
        Sort a list of CPU features ordered by the lowest interest.

        Parameters
        ----------
        'names': sequence
            sequence of supported feature names in uppercase.
        'reverse': bool, optional
            If true, the sorted features is reversed. (highest interest)

        Returns
        -------
        list, sorted CPU features
        """

        def sort_cb(k):
            if isinstance(k, str):
                return self.feature_supported[k]['interest']
            rank = max([self.feature_supported[f]['interest'] for f in k])
            rank += len(k) - 1
            return rank
        return sorted(names, reverse=reverse, key=sort_cb)

    def feature_implies(self, names, keep_origins=False):
        """
        Return a set of CPU features that implied by 'names'

        Parameters
        ----------
        names : str or sequence of str
            CPU feature name(s) in uppercase.

        keep_origins : bool
            if False(default) then the returned set will not contain any
            features from 'names'. This case happens only when two features
            imply each other.

        Examples
        --------
        >>> self.feature_implies("SSE3")
        {'SSE', 'SSE2'}
        >>> self.feature_implies("SSE2")
        {'SSE'}
        >>> self.feature_implies("SSE2", keep_origins=True)
        # 'SSE2' found here since 'SSE' and 'SSE2' imply each other
        {'SSE', 'SSE2'}
        """

        def get_implies(name, _caller=set()):
            implies = set()
            d = self.feature_supported[name]
            for i in d.get('implies', []):
                implies.add(i)
                if i in _caller:
                    continue
                _caller.add(name)
                implies = implies.union(get_implies(i, _caller))
            return implies
        if isinstance(names, str):
            implies = get_implies(names)
            names = [names]
        else:
            assert hasattr(names, '__iter__')
            implies = set()
            for n in names:
                implies = implies.union(get_implies(n))
        if not keep_origins:
            implies.difference_update(names)
        return implies

    def feature_implies_c(self, names):
        """same as feature_implies() but combining 'names'"""
        if isinstance(names, str):
            names = set((names,))
        else:
            names = set(names)
        return names.union(self.feature_implies(names))

    def feature_ahead(self, names):
        """
        Return list of features in 'names' after remove any
        implied features and keep the origins.

        Parameters
        ----------
        'names': sequence
            sequence of CPU feature names in uppercase.

        Returns
        -------
        list of CPU features sorted as-is 'names'

        Examples
        --------
        >>> self.feature_ahead(["SSE2", "SSE3", "SSE41"])
        ["SSE41"]
        # assume AVX2 and FMA3 implies each other and AVX2
        # is the highest interest
        >>> self.feature_ahead(["SSE2", "SSE3", "SSE41", "AVX2", "FMA3"])
        ["AVX2"]
        # assume AVX2 and FMA3 don't implies each other
        >>> self.feature_ahead(["SSE2", "SSE3", "SSE41", "AVX2", "FMA3"])
        ["AVX2", "FMA3"]
        """
        assert not isinstance(names, str) and hasattr(names, '__iter__')
        implies = self.feature_implies(names, keep_origins=True)
        ahead = [n for n in names if n not in implies]
        if len(ahead) == 0:
            ahead = self.feature_sorted(names, reverse=True)[:1]
        return ahead

    def feature_untied(self, names):
        """
        same as 'feature_ahead()' but if both features implied each other
        and keep the highest interest.

        Parameters
        ----------
        'names': sequence
            sequence of CPU feature names in uppercase.

        Returns
        -------
        list of CPU features sorted as-is 'names'

        Examples
        --------
        >>> self.feature_untied(["SSE2", "SSE3", "SSE41"])
        ["SSE2", "SSE3", "SSE41"]
        # assume AVX2 and FMA3 implies each other
        >>> self.feature_untied(["SSE2", "SSE3", "SSE41", "FMA3", "AVX2"])
        ["SSE2", "SSE3", "SSE41", "AVX2"]
        """
        assert not isinstance(names, str) and hasattr(names, '__iter__')
        final = []
        for n in names:
            implies = self.feature_implies(n)
            tied = [nn for nn in final if nn in implies and n in self.feature_implies(nn)]
            if tied:
                tied = self.feature_sorted(tied + [n])
                if n not in tied[1:]:
                    continue
                final.remove(tied[:1][0])
            final.append(n)
        return final

    def feature_get_til(self, names, keyisfalse):
        """
        same as `feature_implies_c()` but stop collecting implied
        features when feature's option that provided through
        parameter 'keyisfalse' is False, also sorting the returned
        features.
        """

        def til(tnames):
            tnames = self.feature_implies_c(tnames)
            tnames = self.feature_sorted(tnames, reverse=True)
            for i, n in enumerate(tnames):
                if not self.feature_supported[n].get(keyisfalse, True):
                    tnames = tnames[:i + 1]
                    break
            return tnames
        if isinstance(names, str) or len(names) <= 1:
            names = til(names)
            names.reverse()
            return names
        names = self.feature_ahead(names)
        names = {t for n in names for t in til(n)}
        return self.feature_sorted(names)

    def feature_detect(self, names):
        """
        Return a list of CPU features that required to be detected
        sorted from the lowest to highest interest.
        """
        names = self.feature_get_til(names, 'implies_detect')
        detect = []
        for n in names:
            d = self.feature_supported[n]
            detect += d.get('detect', d.get('group', [n]))
        return detect

    @_Cache.me
    def feature_flags(self, names):
        """
        Return a list of CPU features flags sorted from the lowest
        to highest interest.
        """
        names = self.feature_sorted(self.feature_implies_c(names))
        flags = []
        for n in names:
            d = self.feature_supported[n]
            f = d.get('flags', [])
            if not f or not self.cc_test_flags(f):
                continue
            flags += f
        return self.cc_normalize_flags(flags)

    @_Cache.me
    def feature_test(self, name, force_flags=None, macros=[]):
        """
        Test a certain CPU feature against the compiler through its own
        check file.

        Parameters
        ----------
        name : str
            Supported CPU feature name.

        force_flags : list or None, optional
            If None(default), the returned flags from `feature_flags()`
            will be used.

        macros : list of tuples, optional
            A list of C macro definitions.
        """
        if force_flags is None:
            force_flags = self.feature_flags(name)
        self.dist_log("testing feature '%s' with flags (%s)" % (name, ' '.join(force_flags)))
        test_path = os.path.join(self.conf_check_path, 'cpu_%s.c' % name.lower())
        if not os.path.exists(test_path):
            self.dist_fatal('feature test file is not exist', test_path)
        test = self.dist_test(test_path, force_flags + self.cc_flags['werror'], macros=macros)
        if not test:
            self.dist_log('testing failed', stderr=True)
        return test

    @_Cache.me
    def feature_is_supported(self, name, force_flags=None, macros=[]):
        """
        Check if a certain CPU feature is supported by the platform and compiler.

        Parameters
        ----------
        name : str
            CPU feature name in uppercase.

        force_flags : list or None, optional
            If None(default), default compiler flags for every CPU feature will
            be used during test.

        macros : list of tuples, optional
            A list of C macro definitions.
        """
        assert name.isupper()
        assert force_flags is None or isinstance(force_flags, list)
        supported = name in self.feature_supported
        if supported:
            for impl in self.feature_implies(name):
                if not self.feature_test(impl, force_flags, macros=macros):
                    return False
            if not self.feature_test(name, force_flags, macros=macros):
                return False
        return supported

    @_Cache.me
    def feature_can_autovec(self, name):
        """
        check if the feature can be auto-vectorized by the compiler
        """
        assert isinstance(name, str)
        d = self.feature_supported[name]
        can = d.get('autovec', None)
        if can is None:
            valid_flags = [self.cc_test_flags([f]) for f in d.get('flags', [])]
            can = valid_flags and any(valid_flags)
        return can

    @_Cache.me
    def feature_extra_checks(self, name):
        """
        Return a list of supported extra checks after testing them against
        the compiler.

        Parameters
        ----------
        names : str
            CPU feature name in uppercase.
        """
        assert isinstance(name, str)
        d = self.feature_supported[name]
        extra_checks = d.get('extra_checks', [])
        if not extra_checks:
            return []
        self.dist_log("Testing extra checks for feature '%s'" % name, extra_checks)
        flags = self.feature_flags(name)
        available = []
        not_available = []
        for chk in extra_checks:
            test_path = os.path.join(self.conf_check_path, 'extra_%s.c' % chk.lower())
            if not os.path.exists(test_path):
                self.dist_fatal('extra check file does not exist', test_path)
            is_supported = self.dist_test(test_path, flags + self.cc_flags['werror'])
            if is_supported:
                available.append(chk)
            else:
                not_available.append(chk)
        if not_available:
            self.dist_log('testing failed for checks', not_available, stderr=True)
        return available

    def feature_c_preprocessor(self, feature_name, tabs=0):
        """
        Generate C preprocessor definitions and include headers of a CPU feature.

        Parameters
        ----------
        'feature_name': str
            CPU feature name in uppercase.
        'tabs': int
            if > 0, align the generated strings to the right depend on number of tabs.

        Returns
        -------
        str, generated C preprocessor

        Examples
        --------
        >>> self.feature_c_preprocessor("SSE3")
        /** SSE3 **/
        #define NPY_HAVE_SSE3 1
        #include <pmmintrin.h>
        """
        assert feature_name.isupper()
        feature = self.feature_supported.get(feature_name)
        assert feature is not None
        prepr = ['/** %s **/' % feature_name, '#define %sHAVE_%s 1' % (self.conf_c_prefix, feature_name)]
        prepr += ['#include <%s>' % h for h in feature.get('headers', [])]
        extra_defs = feature.get('group', [])
        extra_defs += self.feature_extra_checks(feature_name)
        for edef in extra_defs:
            prepr += ['#ifndef %sHAVE_%s' % (self.conf_c_prefix, edef), '\t#define %sHAVE_%s 1' % (self.conf_c_prefix, edef), '#endif']
        if tabs > 0:
            prepr = ['\t' * tabs + l for l in prepr]
        return '\n'.join(prepr)