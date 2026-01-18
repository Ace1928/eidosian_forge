from __future__ import annotations
import dataclasses
import glob
import importlib
import itertools
import json
import os
import shutil
import collections
import typing as T
from . import builder
from . import version
from ..mesonlib import MesonException, Popen_safe, OptionKey
from .. import coredata
def _load_manifests(subdir: str) -> T.Dict[str, Manifest]:
    filename = os.path.join(subdir, 'Cargo.toml')
    raw = load_toml(filename)
    manifests: T.Dict[str, Manifest] = {}
    raw_manifest: T.Union[manifest.Manifest, manifest.VirtualManifest]
    if 'package' in raw:
        raw_manifest = T.cast('manifest.Manifest', raw)
        manifest_ = _convert_manifest(raw_manifest, subdir)
        manifests[manifest_.package.name] = manifest_
    else:
        raw_manifest = T.cast('manifest.VirtualManifest', raw)
    if 'workspace' in raw_manifest:
        pwd = os.getcwd()
        os.chdir(subdir)
        members: T.Iterable[str]
        try:
            members = itertools.chain.from_iterable((glob.glob(m) for m in raw_manifest['workspace']['members']))
        finally:
            os.chdir(pwd)
        if 'exclude' in raw_manifest['workspace']:
            members = (x for x in members if x not in raw_manifest['workspace']['exclude'])
        for m in members:
            filename = os.path.join(subdir, m, 'Cargo.toml')
            raw = load_toml(filename)
            raw_manifest = T.cast('manifest.Manifest', raw)
            man = _convert_manifest(raw_manifest, subdir, m)
            manifests[man.package.name] = man
    return manifests