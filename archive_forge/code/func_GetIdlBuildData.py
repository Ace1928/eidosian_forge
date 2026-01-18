import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def GetIdlBuildData(self, source, config):
    """Determine the implicit outputs for an idl file. Returns output
        directory, outputs, and variables and flags that are required."""
    config = self._TargetConfig(config)
    midl_get = self._GetWrapper(self, self.msvs_settings[config], 'VCMIDLTool')

    def midl(name, default=None):
        return self.ConvertVSMacros(midl_get(name, default=default), config=config)
    tlb = midl('TypeLibraryName', default='${root}.tlb')
    header = midl('HeaderFileName', default='${root}.h')
    dlldata = midl('DLLDataFileName', default='dlldata.c')
    iid = midl('InterfaceIdentifierFileName', default='${root}_i.c')
    proxy = midl('ProxyFileName', default='${root}_p.c')
    outdir = midl('OutputDirectory', default='')
    output = [header, dlldata, iid, proxy]
    variables = [('tlb', tlb), ('h', header), ('dlldata', dlldata), ('iid', iid), ('proxy', proxy)]
    target_platform = self.GetArch(config)
    if target_platform == 'x86':
        target_platform = 'win32'
    flags = ['/char', 'signed', '/env', target_platform, '/Oicf']
    return (outdir, output, variables, flags)