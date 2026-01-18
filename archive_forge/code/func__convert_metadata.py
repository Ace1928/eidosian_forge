import email
import itertools
import functools
import os
import posixpath
import re
import zipfile
import contextlib
from distutils.util import get_platform
import setuptools
from setuptools.extern.packaging.version import Version as parse_version
from setuptools.extern.packaging.tags import sys_tags
from setuptools.extern.packaging.utils import canonicalize_name
from setuptools.command.egg_info import write_requirements, _egg_basename
from setuptools.archive_util import _unpack_zipfile_obj
@staticmethod
def _convert_metadata(zf, destination_eggdir, dist_info, egg_info):
    import pkg_resources

    def get_metadata(name):
        with zf.open(posixpath.join(dist_info, name)) as fp:
            value = fp.read().decode('utf-8')
            return email.parser.Parser().parsestr(value)
    wheel_metadata = get_metadata('WHEEL')
    wheel_version = parse_version(wheel_metadata.get('Wheel-Version'))
    wheel_v1 = parse_version('1.0') <= wheel_version < parse_version('2.0dev0')
    if not wheel_v1:
        raise ValueError('unsupported wheel format version: %s' % wheel_version)
    _unpack_zipfile_obj(zf, destination_eggdir)
    dist_info = os.path.join(destination_eggdir, dist_info)
    dist = pkg_resources.Distribution.from_location(destination_eggdir, dist_info, metadata=pkg_resources.PathMetadata(destination_eggdir, dist_info))

    def raw_req(req):
        req.marker = None
        return str(req)
    install_requires = list(map(raw_req, dist.requires()))
    extras_require = {extra: [req for req in map(raw_req, dist.requires((extra,))) if req not in install_requires] for extra in dist.extras}
    os.rename(dist_info, egg_info)
    os.rename(os.path.join(egg_info, 'METADATA'), os.path.join(egg_info, 'PKG-INFO'))
    setup_dist = setuptools.Distribution(attrs=dict(install_requires=install_requires, extras_require=extras_require))
    with disable_info_traces():
        write_requirements(setup_dist.get_command_obj('egg_info'), None, os.path.join(egg_info, 'requires.txt'))