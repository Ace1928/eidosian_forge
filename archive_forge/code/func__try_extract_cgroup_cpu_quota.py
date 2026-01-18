from __future__ import annotations
import math
import os
import sys
def _try_extract_cgroup_cpu_quota():
    for dirname in ['cpuacct,cpu', 'cpu,cpuacct']:
        try:
            with open('/sys/fs/cgroup/%s/cpu.cfs_quota_us' % dirname) as f:
                quota = int(f.read())
            with open('/sys/fs/cgroup/%s/cpu.cfs_period_us' % dirname) as f:
                period = int(f.read())
            return (quota, period)
        except Exception:
            pass
    try:
        with open('/proc/self/cgroup') as f:
            group_path = f.read().strip().split(':')[-1]
        if not group_path.endswith('/'):
            group_path = f'{group_path}/'
        with open('/sys/fs/cgroup%scpu.max' % group_path) as f:
            quota, period = map(int, f.read().split(' '))
            return (quota, period)
    except Exception:
        pass
    return (None, None)