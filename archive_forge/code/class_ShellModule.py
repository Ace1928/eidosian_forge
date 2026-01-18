from __future__ import (absolute_import, division, print_function)
import shlex
from ansible.plugins.shell import ShellBase
class ShellModule(ShellBase):
    COMPATIBLE_SHELLS = frozenset(('sh', 'zsh', 'bash', 'dash', 'ksh'))
    SHELL_FAMILY = 'sh'
    ECHO = 'echo'
    COMMAND_SEP = ';'
    _SHELL_EMBEDDED_PY_EOL = '\n'
    _SHELL_REDIRECT_ALLNULL = '> /dev/null 2>&1'
    _SHELL_AND = '&&'
    _SHELL_OR = '||'
    _SHELL_SUB_LEFT = '"`'
    _SHELL_SUB_RIGHT = '`"'
    _SHELL_GROUP_LEFT = '('
    _SHELL_GROUP_RIGHT = ')'

    def checksum(self, path, python_interp):
        shell_escaped_path = shlex.quote(path)
        test = 'rc=flag; [ -r %(p)s ] %(shell_or)s rc=2; [ -f %(p)s ] %(shell_or)s rc=1; [ -d %(p)s ] %(shell_and)s rc=3; %(i)s -V 2>/dev/null %(shell_or)s rc=4; [ x"$rc" != "xflag" ] %(shell_and)s echo "${rc}  "%(p)s %(shell_and)s exit 0' % dict(p=shell_escaped_path, i=python_interp, shell_and=self._SHELL_AND, shell_or=self._SHELL_OR)
        csums = [u'({0} -c \'import hashlib; BLOCKSIZE = 65536; hasher = hashlib.sha1();{2}afile = open("\'{1}\'", "rb"){2}buf = afile.read(BLOCKSIZE){2}while len(buf) > 0:{2}\thasher.update(buf){2}\tbuf = afile.read(BLOCKSIZE){2}afile.close(){2}print(hasher.hexdigest())\' 2>/dev/null)'.format(python_interp, shell_escaped_path, self._SHELL_EMBEDDED_PY_EOL), u'({0} -c \'import sha; BLOCKSIZE = 65536; hasher = sha.sha();{2}afile = open("\'{1}\'", "rb"){2}buf = afile.read(BLOCKSIZE){2}while len(buf) > 0:{2}\thasher.update(buf){2}\tbuf = afile.read(BLOCKSIZE){2}afile.close(){2}print(hasher.hexdigest())\' 2>/dev/null)'.format(python_interp, shell_escaped_path, self._SHELL_EMBEDDED_PY_EOL)]
        cmd = (' %s ' % self._SHELL_OR).join(csums)
        cmd = "%s; %s %s (echo '0  '%s)" % (test, cmd, self._SHELL_OR, shell_escaped_path)
        return cmd