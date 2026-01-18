from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.general.plugins.module_utils.cmd_runner import CmdRunner, cmd_runner_fmt as fmt
from ansible_collections.community.general.plugins.module_utils.module_helper import ModuleHelper, ModuleHelperException
def _list_element(self, _type, path_re, elem_re):

    def process(rc, out, err):
        return [] if 'None of the provided paths were usable' in out else out.splitlines()
    with self.runner('type galaxy_cmd dest', output_process=process, check_rc=False) as ctx:
        elems = ctx.run(type=_type, galaxy_cmd='list')
    elems_dict = {}
    current_path = None
    for line in elems:
        if line.startswith('#'):
            match = path_re.match(line)
            if not match:
                continue
            if self.vars.dest is not None and match.group('path') != self.vars.dest:
                current_path = None
                continue
            current_path = match.group('path') if match else None
            elems_dict[current_path] = {}
        elif current_path is not None:
            match = elem_re.match(line)
            if not match or (self.vars.name is not None and match.group('elem') != self.vars.name):
                continue
            elems_dict[current_path][match.group('elem')] = match.group('version')
    return elems_dict