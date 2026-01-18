import errno
import os
import re
import subprocess
import sys
import glob
def _DetectVisualStudioVersions(versions_to_check, force_express):
    """Collect the list of installed visual studio versions.

  Returns:
    A list of visual studio versions installed in descending order of
    usage preference.
    Base this on the registry and a quick check if devenv.exe exists.
    Possibilities are:
      2005(e) - Visual Studio 2005 (8)
      2008(e) - Visual Studio 2008 (9)
      2010(e) - Visual Studio 2010 (10)
      2012(e) - Visual Studio 2012 (11)
      2013(e) - Visual Studio 2013 (12)
      2015    - Visual Studio 2015 (14)
      2017    - Visual Studio 2017 (15)
      2019    - Visual Studio 2019 (16)
    Where (e) is e for express editions of MSVS and blank otherwise.
  """
    version_to_year = {'8.0': '2005', '9.0': '2008', '10.0': '2010', '11.0': '2012', '12.0': '2013', '14.0': '2015', '15.0': '2017', '16.0': '2019'}
    versions = []
    for version in versions_to_check:
        keys = ['HKLM\\Software\\Microsoft\\VisualStudio\\%s' % version, 'HKLM\\Software\\Wow6432Node\\Microsoft\\VisualStudio\\%s' % version, 'HKLM\\Software\\Microsoft\\VCExpress\\%s' % version, 'HKLM\\Software\\Wow6432Node\\Microsoft\\VCExpress\\%s' % version]
        for index in range(len(keys)):
            path = _RegistryGetValue(keys[index], 'InstallDir')
            if not path:
                continue
            path = _ConvertToCygpath(path)
            full_path = os.path.join(path, 'devenv.exe')
            express_path = os.path.join(path, '*express.exe')
            if not force_express and os.path.exists(full_path):
                versions.append(_CreateVersion(version_to_year[version], os.path.join(path, '..', '..')))
            elif glob.glob(express_path):
                versions.append(_CreateVersion(version_to_year[version] + 'e', os.path.join(path, '..', '..')))
        keys = ['HKLM\\Software\\Microsoft\\VisualStudio\\SxS\\VC7', 'HKLM\\Software\\Wow6432Node\\Microsoft\\VisualStudio\\SxS\\VC7', 'HKLM\\Software\\Microsoft\\VisualStudio\\SxS\\VS7', 'HKLM\\Software\\Wow6432Node\\Microsoft\\VisualStudio\\SxS\\VS7']
        for index in range(len(keys)):
            path = _RegistryGetValue(keys[index], version)
            if not path:
                continue
            path = _ConvertToCygpath(path)
            if version == '15.0':
                if os.path.exists(path):
                    versions.append(_CreateVersion('2017', path))
            elif version != '14.0':
                versions.append(_CreateVersion(version_to_year[version] + 'e', os.path.join(path, '..'), sdk_based=True))
    return versions