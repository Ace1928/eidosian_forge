from __future__ import absolute_import, division, print_function
import re
import socket
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def add_bricks(name, new_bricks, stripe, replica, force):
    args = ['volume', 'add-brick', name]
    if stripe:
        args.append('stripe')
        args.append(str(stripe))
    if replica:
        args.append('replica')
        args.append(str(replica))
    args.extend(new_bricks)
    if force:
        args.append('force')
    run_gluster(args)