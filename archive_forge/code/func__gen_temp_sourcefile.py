import os, re
from distutils.core import Command
from distutils.errors import DistutilsExecError
from distutils.sysconfig import customize_compiler
from distutils import log
def _gen_temp_sourcefile(self, body, headers, lang):
    filename = '_configtest' + LANG_EXT[lang]
    with open(filename, 'w') as file:
        if headers:
            for header in headers:
                file.write('#include <%s>\n' % header)
            file.write('\n')
        file.write(body)
        if body[-1] != '\n':
            file.write('\n')
    return filename