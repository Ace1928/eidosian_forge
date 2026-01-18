import sys
import os
import io
import time
import re
import types
from typing import Protocol
import zipfile
import zipimport
import warnings
import stat
import functools
import pkgutil
import operator
import platform
import collections
import plistlib
import email.parser
import errno
import tempfile
import textwrap
import inspect
import ntpath
import posixpath
import importlib
import importlib.machinery
from pkgutil import get_importer
import _imp
from os import utime
from os import open as os_open
from os.path import isdir, split
from pkg_resources.extern.jaraco.text import (
from pkg_resources.extern import platformdirs
from pkg_resources.extern import packaging
class WorkingSet:
    """A collection of active distributions on sys.path (or a similar list)"""

    def __init__(self, entries=None):
        """Create working set from list of path entries (default=sys.path)"""
        self.entries = []
        self.entry_keys = {}
        self.by_key = {}
        self.normalized_to_canonical_keys = {}
        self.callbacks = []
        if entries is None:
            entries = sys.path
        for entry in entries:
            self.add_entry(entry)

    @classmethod
    def _build_master(cls):
        """
        Prepare the master working set.
        """
        ws = cls()
        try:
            from __main__ import __requires__
        except ImportError:
            return ws
        try:
            ws.require(__requires__)
        except VersionConflict:
            return cls._build_from_requirements(__requires__)
        return ws

    @classmethod
    def _build_from_requirements(cls, req_spec):
        """
        Build a working set from a requirement spec. Rewrites sys.path.
        """
        ws = cls([])
        reqs = parse_requirements(req_spec)
        dists = ws.resolve(reqs, Environment())
        for dist in dists:
            ws.add(dist)
        for entry in sys.path:
            if entry not in ws.entries:
                ws.add_entry(entry)
        sys.path[:] = ws.entries
        return ws

    def add_entry(self, entry):
        """Add a path item to ``.entries``, finding any distributions on it

        ``find_distributions(entry, True)`` is used to find distributions
        corresponding to the path entry, and they are added.  `entry` is
        always appended to ``.entries``, even if it is already present.
        (This is because ``sys.path`` can contain the same value more than
        once, and the ``.entries`` of the ``sys.path`` WorkingSet should always
        equal ``sys.path``.)
        """
        self.entry_keys.setdefault(entry, [])
        self.entries.append(entry)
        for dist in find_distributions(entry, True):
            self.add(dist, entry, False)

    def __contains__(self, dist):
        """True if `dist` is the active distribution for its project"""
        return self.by_key.get(dist.key) == dist

    def find(self, req):
        """Find a distribution matching requirement `req`

        If there is an active distribution for the requested project, this
        returns it as long as it meets the version requirement specified by
        `req`.  But, if there is an active distribution for the project and it
        does *not* meet the `req` requirement, ``VersionConflict`` is raised.
        If there is no active distribution for the requested project, ``None``
        is returned.
        """
        dist = self.by_key.get(req.key)
        if dist is None:
            canonical_key = self.normalized_to_canonical_keys.get(req.key)
            if canonical_key is not None:
                req.key = canonical_key
                dist = self.by_key.get(canonical_key)
        if dist is not None and dist not in req:
            raise VersionConflict(dist, req)
        return dist

    def iter_entry_points(self, group, name=None):
        """Yield entry point objects from `group` matching `name`

        If `name` is None, yields all entry points in `group` from all
        distributions in the working set, otherwise only ones matching
        both `group` and `name` are yielded (in distribution order).
        """
        return (entry for dist in self for entry in dist.get_entry_map(group).values() if name is None or name == entry.name)

    def run_script(self, requires, script_name):
        """Locate distribution for `requires` and run `script_name` script"""
        ns = sys._getframe(1).f_globals
        name = ns['__name__']
        ns.clear()
        ns['__name__'] = name
        self.require(requires)[0].run_script(script_name, ns)

    def __iter__(self):
        """Yield distributions for non-duplicate projects in the working set

        The yield order is the order in which the items' path entries were
        added to the working set.
        """
        seen = {}
        for item in self.entries:
            if item not in self.entry_keys:
                continue
            for key in self.entry_keys[item]:
                if key not in seen:
                    seen[key] = 1
                    yield self.by_key[key]

    def add(self, dist, entry=None, insert=True, replace=False):
        """Add `dist` to working set, associated with `entry`

        If `entry` is unspecified, it defaults to the ``.location`` of `dist`.
        On exit from this routine, `entry` is added to the end of the working
        set's ``.entries`` (if it wasn't already present).

        `dist` is only added to the working set if it's for a project that
        doesn't already have a distribution in the set, unless `replace=True`.
        If it's added, any callbacks registered with the ``subscribe()`` method
        will be called.
        """
        if insert:
            dist.insert_on(self.entries, entry, replace=replace)
        if entry is None:
            entry = dist.location
        keys = self.entry_keys.setdefault(entry, [])
        keys2 = self.entry_keys.setdefault(dist.location, [])
        if not replace and dist.key in self.by_key:
            return
        self.by_key[dist.key] = dist
        normalized_name = packaging.utils.canonicalize_name(dist.key)
        self.normalized_to_canonical_keys[normalized_name] = dist.key
        if dist.key not in keys:
            keys.append(dist.key)
        if dist.key not in keys2:
            keys2.append(dist.key)
        self._added_new(dist)

    def resolve(self, requirements, env=None, installer=None, replace_conflicting=False, extras=None):
        """List all distributions needed to (recursively) meet `requirements`

        `requirements` must be a sequence of ``Requirement`` objects.  `env`,
        if supplied, should be an ``Environment`` instance.  If
        not supplied, it defaults to all distributions available within any
        entry or distribution in the working set.  `installer`, if supplied,
        will be invoked with each requirement that cannot be met by an
        already-installed distribution; it should return a ``Distribution`` or
        ``None``.

        Unless `replace_conflicting=True`, raises a VersionConflict exception
        if
        any requirements are found on the path that have the correct name but
        the wrong version.  Otherwise, if an `installer` is supplied it will be
        invoked to obtain the correct version of the requirement and activate
        it.

        `extras` is a list of the extras to be used with these requirements.
        This is important because extra requirements may look like `my_req;
        extra = "my_extra"`, which would otherwise be interpreted as a purely
        optional requirement.  Instead, we want to be able to assert that these
        requirements are truly required.
        """
        requirements = list(requirements)[::-1]
        processed = {}
        best = {}
        to_activate = []
        req_extras = _ReqExtras()
        required_by = collections.defaultdict(set)
        while requirements:
            req = requirements.pop(0)
            if req in processed:
                continue
            if not req_extras.markers_pass(req, extras):
                continue
            dist = self._resolve_dist(req, best, replace_conflicting, env, installer, required_by, to_activate)
            new_requirements = dist.requires(req.extras)[::-1]
            requirements.extend(new_requirements)
            for new_requirement in new_requirements:
                required_by[new_requirement].add(req.project_name)
                req_extras[new_requirement] = req.extras
            processed[req] = True
        return to_activate

    def _resolve_dist(self, req, best, replace_conflicting, env, installer, required_by, to_activate):
        dist = best.get(req.key)
        if dist is None:
            dist = self.by_key.get(req.key)
            if dist is None or (dist not in req and replace_conflicting):
                ws = self
                if env is None:
                    if dist is None:
                        env = Environment(self.entries)
                    else:
                        env = Environment([])
                        ws = WorkingSet([])
                dist = best[req.key] = env.best_match(req, ws, installer, replace_conflicting=replace_conflicting)
                if dist is None:
                    requirers = required_by.get(req, None)
                    raise DistributionNotFound(req, requirers)
            to_activate.append(dist)
        if dist not in req:
            dependent_req = required_by[req]
            raise VersionConflict(dist, req).with_context(dependent_req)
        return dist

    def find_plugins(self, plugin_env, full_env=None, installer=None, fallback=True):
        """Find all activatable distributions in `plugin_env`

        Example usage::

            distributions, errors = working_set.find_plugins(
                Environment(plugin_dirlist)
            )
            # add plugins+libs to sys.path
            map(working_set.add, distributions)
            # display errors
            print('Could not load', errors)

        The `plugin_env` should be an ``Environment`` instance that contains
        only distributions that are in the project's "plugin directory" or
        directories. The `full_env`, if supplied, should be an ``Environment``
        contains all currently-available distributions.  If `full_env` is not
        supplied, one is created automatically from the ``WorkingSet`` this
        method is called on, which will typically mean that every directory on
        ``sys.path`` will be scanned for distributions.

        `installer` is a standard installer callback as used by the
        ``resolve()`` method. The `fallback` flag indicates whether we should
        attempt to resolve older versions of a plugin if the newest version
        cannot be resolved.

        This method returns a 2-tuple: (`distributions`, `error_info`), where
        `distributions` is a list of the distributions found in `plugin_env`
        that were loadable, along with any other distributions that are needed
        to resolve their dependencies.  `error_info` is a dictionary mapping
        unloadable plugin distributions to an exception instance describing the
        error that occurred. Usually this will be a ``DistributionNotFound`` or
        ``VersionConflict`` instance.
        """
        plugin_projects = list(plugin_env)
        plugin_projects.sort()
        error_info = {}
        distributions = {}
        if full_env is None:
            env = Environment(self.entries)
            env += plugin_env
        else:
            env = full_env + plugin_env
        shadow_set = self.__class__([])
        list(map(shadow_set.add, self))
        for project_name in plugin_projects:
            for dist in plugin_env[project_name]:
                req = [dist.as_requirement()]
                try:
                    resolvees = shadow_set.resolve(req, env, installer)
                except ResolutionError as v:
                    error_info[dist] = v
                    if fallback:
                        continue
                    else:
                        break
                else:
                    list(map(shadow_set.add, resolvees))
                    distributions.update(dict.fromkeys(resolvees))
                    break
        distributions = list(distributions)
        distributions.sort()
        return (distributions, error_info)

    def require(self, *requirements):
        """Ensure that distributions matching `requirements` are activated

        `requirements` must be a string or a (possibly-nested) sequence
        thereof, specifying the distributions and versions required.  The
        return value is a sequence of the distributions that needed to be
        activated to fulfill the requirements; all relevant distributions are
        included, even if they were already activated in this working set.
        """
        needed = self.resolve(parse_requirements(requirements))
        for dist in needed:
            self.add(dist)
        return needed

    def subscribe(self, callback, existing=True):
        """Invoke `callback` for all distributions

        If `existing=True` (default),
        call on all existing ones, as well.
        """
        if callback in self.callbacks:
            return
        self.callbacks.append(callback)
        if not existing:
            return
        for dist in self:
            callback(dist)

    def _added_new(self, dist):
        for callback in self.callbacks:
            callback(dist)

    def __getstate__(self):
        return (self.entries[:], self.entry_keys.copy(), self.by_key.copy(), self.normalized_to_canonical_keys.copy(), self.callbacks[:])

    def __setstate__(self, e_k_b_n_c):
        entries, keys, by_key, normalized_to_canonical_keys, callbacks = e_k_b_n_c
        self.entries = entries[:]
        self.entry_keys = keys.copy()
        self.by_key = by_key.copy()
        self.normalized_to_canonical_keys = normalized_to_canonical_keys.copy()
        self.callbacks = callbacks[:]