import os
from configparser import ConfigParser
from distutils import log
from distutils.command.build_py import build_py
from distutils.command.install_scripts import install_scripts
from distutils.version import LooseVersion
from os.path import join as pjoin
from os.path import split as psplit
from os.path import splitext
class install_scripts_bat(install_scripts):
    """Make scripts executable on Windows

    Scripts are bare file names without extension on Unix, fitting (for example)
    Debian rules. They identify as python scripts with the usual ``#!`` first
    line. Unix recognizes and uses this first "shebang" line, but Windows does
    not. So, on Windows only we add a ``.bat`` wrapper of name
    ``bare_script_name.bat`` to call ``bare_script_name`` using the python
    interpreter from the #! first line of the script.

    Notes
    -----
    See discussion at
    https://matthew-brett.github.io/pydagogue/installing_scripts.html and
    example at git://github.com/matthew-brett/myscripter.git for more
    background.
    """

    def run(self):
        install_scripts.run(self)
        if not os.name == 'nt':
            return
        for filepath in self.get_outputs():
            with open(filepath, 'rt') as fobj:
                first_line = fobj.readline()
            if not (first_line.startswith('#!') and 'python' in first_line.lower()):
                log.info('No #!python executable found, skipping .bat wrapper')
                continue
            pth, fname = psplit(filepath)
            froot, ext = splitext(fname)
            bat_file = pjoin(pth, froot + '.bat')
            bat_contents = BAT_TEMPLATE.replace('{FNAME}', fname)
            log.info(f'Making {bat_file} wrapper for {filepath}')
            if self.dry_run:
                continue
            with open(bat_file, 'wt') as fobj:
                fobj.write(bat_contents)