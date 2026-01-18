from __future__ import (absolute_import, division, print_function)
import os
import re
import stat
from ansible.module_utils.basic import AnsibleModule
def _add_pbkdf_options(self, options, pbkdf):
    if pbkdf['iteration_time'] is not None:
        options.extend(['--iter-time', str(int(pbkdf['iteration_time'] * 1000))])
    if pbkdf['iteration_count'] is not None:
        options.extend(['--pbkdf-force-iterations', str(pbkdf['iteration_count'])])
    if pbkdf['algorithm'] is not None:
        options.extend(['--pbkdf', pbkdf['algorithm']])
    if pbkdf['memory'] is not None:
        options.extend(['--pbkdf-memory', str(pbkdf['memory'])])
    if pbkdf['parallel'] is not None:
        options.extend(['--pbkdf-parallel', str(pbkdf['parallel'])])