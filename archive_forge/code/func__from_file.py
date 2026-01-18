import os
import collections.abc
@classmethod
def _from_file(cls, tuple_table_fd, cpu_table_fd, triplet_compat=False):
    arch2tuple = {}
    cpu_list = [x[0] for x in _parse_table_file(cpu_table_fd)]
    for row in _parse_table_file(tuple_table_fd):
        dpkg_tuple = row[0]
        dpkg_arch = row[1]
        if triplet_compat:
            dpkg_tuple = 'base-' + dpkg_tuple
        if '<cpu>' in dpkg_tuple:
            for cpu_name in cpu_list:
                debtuple_cpu = dpkg_tuple.replace('<cpu>', cpu_name)
                dpkg_arch_cpu = dpkg_arch.replace('<cpu>', cpu_name)
                arch2tuple[dpkg_arch_cpu] = QuadTupleDpkgArchitecture(*debtuple_cpu.split('-', 3))
        else:
            arch2tuple[dpkg_arch] = QuadTupleDpkgArchitecture(*dpkg_tuple.split('-', 3))
    return DpkgArchTable(arch2tuple)