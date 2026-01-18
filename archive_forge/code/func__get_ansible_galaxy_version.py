from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.general.plugins.module_utils.cmd_runner import CmdRunner, cmd_runner_fmt as fmt
from ansible_collections.community.general.plugins.module_utils.module_helper import ModuleHelper, ModuleHelperException
def _get_ansible_galaxy_version(self):

    class UnsupportedLocale(ModuleHelperException):
        pass

    def process(rc, out, err):
        if rc != 0 and 'unsupported locale setting' in err or (rc == 0 and 'cannot change locale' in err):
            raise UnsupportedLocale(msg=err)
        line = out.splitlines()[0]
        match = self._RE_GALAXY_VERSION.match(line)
        if not match:
            self.do_raise('Unable to determine ansible-galaxy version from: {0}'.format(line))
        version = match.group('version')
        version = tuple((int(x) for x in version.split('.')[:3]))
        return version
    try:
        runner = self._make_runner('C.UTF-8')
        with runner('version', check_rc=False, output_process=process) as ctx:
            return (runner, ctx.run(version=True))
    except UnsupportedLocale as e:
        runner = self._make_runner('en_US.UTF-8')
        with runner('version', check_rc=True, output_process=process) as ctx:
            return (runner, ctx.run(version=True))