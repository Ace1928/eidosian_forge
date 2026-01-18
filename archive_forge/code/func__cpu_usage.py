import logging
from ray._private.utils import get_num_cpus
def _cpu_usage():
    """Compute total cpu usage of the container in nanoseconds
    by reading from cpuacct in cgroups v1 or cpu.stat in cgroups v2."""
    try:
        return int(open(CPU_USAGE_PATH).read())
    except FileNotFoundError:
        cpu_stat_text = open(CPU_USAGE_PATH_V2).read()
        cpu_stat_first_line = cpu_stat_text.split('\n')[0]
        cpu_usec = int(cpu_stat_first_line.split()[1])
        return cpu_usec * 1000