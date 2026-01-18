from __future__ import absolute_import, division, print_function
import errno
import os
import platform
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.posix.plugins.module_utils.mount import ismount
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.parsing.convert_bool import boolean
def _set_mount_save_old(module, args):
    """Set/change a mount point location in fstab. Save the old fstab contents."""
    to_write = []
    old_lines = []
    exists = False
    changed = False
    escaped_args = dict([(k, _escape_fstab(v)) for k, v in iteritems(args) if k != 'warnings'])
    new_line = '%(src)s %(name)s %(fstype)s %(opts)s %(dump)s %(passno)s\n'
    if platform.system() == 'SunOS':
        new_line = '%(src)s - %(name)s %(fstype)s %(passno)s %(boot)s %(opts)s\n'
    for line in open(args['fstab'], 'r').readlines():
        if not line.endswith('\n'):
            line += '\n'
        old_lines.append(line)
        if not line.strip():
            to_write.append(line)
            continue
        if line.strip().startswith('#'):
            to_write.append(line)
            continue
        fields = line.split()
        if platform.system() == 'SunOS' and len(fields) != 7 or (platform.system() == 'Linux' and len(fields) not in [4, 5, 6]) or (platform.system() not in ['SunOS', 'Linux'] and len(fields) != 6):
            to_write.append(line)
            continue
        ld = {}
        if platform.system() == 'SunOS':
            ld['src'], dash, ld['name'], ld['fstype'], ld['passno'], ld['boot'], ld['opts'] = fields
        else:
            fields_labels = ['src', 'name', 'fstype', 'opts', 'dump', 'passno']
            ld['dump'] = 0
            ld['passno'] = 0
            for i, field in enumerate(fields):
                ld[fields_labels[i]] = field
        if ld['name'] != escaped_args['name'] or ('src' in args and ld['name'] == 'none' and (ld['fstype'] == 'swap') and (ld['src'] != args['src'])):
            to_write.append(line)
            continue
        exists = True
        args_to_check = ('src', 'fstype', 'opts', 'dump', 'passno')
        if platform.system() == 'SunOS':
            args_to_check = ('src', 'fstype', 'passno', 'boot', 'opts')
        for t in args_to_check:
            if ld[t] != escaped_args[t]:
                ld[t] = escaped_args[t]
                changed = True
        if changed:
            to_write.append(new_line % ld)
        else:
            to_write.append(line)
    if not exists:
        to_write.append(new_line % escaped_args)
        changed = True
    if changed and (not module.check_mode):
        args['backup_file'] = write_fstab(module, to_write, args['fstab'])
    return (args['name'], old_lines, changed)