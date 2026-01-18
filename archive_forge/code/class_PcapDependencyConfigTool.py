from __future__ import annotations
import functools
import re
import typing as T
from .. import mesonlib
from .. import mlog
from .base import DependencyException, DependencyMethods
from .base import BuiltinDependency, SystemDependency
from .cmake import CMakeDependency, CMakeDependencyFactory
from .configtool import ConfigToolDependency
from .detect import packages
from .factory import DependencyFactory, factory_methods
from .pkgconfig import PkgConfigDependency
class PcapDependencyConfigTool(ConfigToolDependency):
    tools = ['pcap-config']
    tool_name = 'pcap-config'
    skip_version = '--help'

    def __init__(self, name: str, environment: 'Environment', kwargs: T.Dict[str, T.Any]):
        super().__init__(name, environment, kwargs)
        if not self.is_found:
            return
        self.compile_args = self.get_config_value(['--cflags'], 'compile_args')
        self.link_args = self.get_config_value(['--libs'], 'link_args')
        if self.version is None:
            self.version = self.get_pcap_lib_version()

    def get_pcap_lib_version(self) -> T.Optional[str]:
        if not self.env.machines.matches_build_machine(self.for_machine):
            return None
        v = self.clib_compiler.get_return_value('pcap_lib_version', 'string', '#include <pcap.h>', self.env, [], [self])
        v = re.sub('libpcap version ', '', str(v))
        v = re.sub(' -- Apple version.*$', '', v)
        return v