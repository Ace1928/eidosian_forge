import os
import subprocess as sp
import shlex
import simplejson as json
from traits.trait_errors import TraitError
from ... import config, logging, LooseVersion
from ...utils.provenance import write_provenance
from ...utils.misc import str2bool
from ...utils.filemanip import (
from ...utils.subprocess import run_command
from ...external.due import due
from .traits_extension import traits, isdefined, Undefined
from .specs import (
from .support import (
class PackageInfo(object):
    _version = None
    version_cmd = None
    version_file = None

    @classmethod
    def version(klass):
        if klass._version is None:
            if klass.version_cmd is not None:
                try:
                    clout = CommandLine(command=klass.version_cmd, resource_monitor=False, terminal_output='allatonce').run()
                except IOError:
                    return None
                raw_info = clout.runtime.stdout
            elif klass.version_file is not None:
                try:
                    with open(klass.version_file, 'rt') as fobj:
                        raw_info = fobj.read()
                except OSError:
                    return None
            else:
                return None
            klass._version = klass.parse_version(raw_info)
        return klass._version

    @staticmethod
    def parse_version(raw_info):
        raise NotImplementedError