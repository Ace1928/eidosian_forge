import os
import re
from stat import ST_MODE
from distutils import sysconfig
from ..core import Command
from .._modified import newer
from ..util import convert_path
from distutils._log import log
import tokenize
def _copy_script(self, script, outfiles, updated_files):
    shebang_match = None
    script = convert_path(script)
    outfile = os.path.join(self.build_dir, os.path.basename(script))
    outfiles.append(outfile)
    if not self.force and (not newer(script, outfile)):
        log.debug('not copying %s (up-to-date)', script)
        return
    try:
        f = tokenize.open(script)
    except OSError:
        if not self.dry_run:
            raise
        f = None
    else:
        first_line = f.readline()
        if not first_line:
            self.warn('%s is an empty file (skipping)' % script)
            return
        shebang_match = shebang_pattern.match(first_line)
    updated_files.append(outfile)
    if shebang_match:
        log.info('copying and adjusting %s -> %s', script, self.build_dir)
        if not self.dry_run:
            if not sysconfig.python_build:
                executable = self.executable
            else:
                executable = os.path.join(sysconfig.get_config_var('BINDIR'), 'python%s%s' % (sysconfig.get_config_var('VERSION'), sysconfig.get_config_var('EXE')))
            post_interp = shebang_match.group(1) or ''
            shebang = '#!' + executable + post_interp + '\n'
            self._validate_shebang(shebang, f.encoding)
            with open(outfile, 'w', encoding=f.encoding) as outf:
                outf.write(shebang)
                outf.writelines(f.readlines())
        if f:
            f.close()
    else:
        if f:
            f.close()
        self.copy_file(script, outfile)