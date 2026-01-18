import glob
import logging
import os
import platform
import re
import subprocess
import sys
from argparse import ArgumentParser, RawTextHelpFormatter, REMAINDER
from os.path import expanduser
from typing import Dict, List
from torch.distributed.elastic.multiprocessing import start_processes, Std
class _Launcher:
    """Class for launcher."""
    msg_lib_notfound = f'Unable to find the {{0}} library file lib{{1}}.so in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or {expanduser('~')}/.local/lib/ so the LD_PRELOAD environment variable will not be set.'

    def __init__(self):
        self.cpuinfo = _CPUinfo()

    def add_lib_preload(self, lib_type):
        """Enable TCMalloc/JeMalloc/intel OpenMP."""
        library_paths = []
        if 'CONDA_PREFIX' in os.environ:
            library_paths.append(f'{os.environ['CONDA_PREFIX']}/lib')
        if 'VIRTUAL_ENV' in os.environ:
            library_paths.append(f'{os.environ['VIRTUAL_ENV']}/lib')
        library_paths += [f'{expanduser('~')}/.local/lib', '/usr/local/lib', '/usr/local/lib64', '/usr/lib', '/usr/lib64']
        lib_find = False
        lib_set = False
        for item in os.getenv('LD_PRELOAD', '').split(':'):
            if item.endswith(f'lib{lib_type}.so'):
                lib_set = True
                break
        if not lib_set:
            for lib_path in library_paths:
                library_file = os.path.join(lib_path, f'lib{lib_type}.so')
                matches = glob.glob(library_file)
                if len(matches) > 0:
                    ld_preloads = [f'{matches[0]}', os.getenv('LD_PRELOAD', '')]
                    os.environ['LD_PRELOAD'] = os.pathsep.join([p.strip(os.pathsep) for p in ld_preloads if p])
                    lib_find = True
                    break
        return lib_set or lib_find

    def is_numactl_available(self):
        numactl_available = False
        try:
            cmd = ['numactl', '-C', '0', '-m', '0', 'hostname']
            r = subprocess.run(cmd, env=os.environ, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            if r.returncode == 0:
                numactl_available = True
        except Exception:
            pass
        return numactl_available

    def set_memory_allocator(self, enable_tcmalloc=True, enable_jemalloc=False, use_default_allocator=False):
        """
        Enable TCMalloc/JeMalloc with LD_PRELOAD and set configuration for JeMalloc.

        By default, PTMalloc will be used for PyTorch, but TCMalloc and JeMalloc can get better
        memory reuse and reduce page fault to improve performance.
        """
        if enable_tcmalloc and enable_jemalloc:
            raise RuntimeError('Unable to enable TCMalloc and JEMalloc at the same time.')
        if enable_tcmalloc:
            find_tc = self.add_lib_preload(lib_type='tcmalloc')
            if not find_tc:
                msg = f'{self.msg_lib_notfound} you can use "conda install -c conda-forge gperftools" to install {{0}}'
                logger.warning(msg.format('TCmalloc', 'tcmalloc'))
            else:
                logger.info('Use TCMalloc memory allocator')
        elif enable_jemalloc:
            find_je = self.add_lib_preload(lib_type='jemalloc')
            if not find_je:
                msg = f'{self.msg_lib_notfound} you can use "conda install -c conda-forge jemalloc" to install {{0}}'
                logger.warning(msg.format('Jemalloc', 'jemalloc'))
            else:
                logger.info('Use JeMalloc memory allocator')
                self.set_env('MALLOC_CONF', 'oversize_threshold:1,background_thread:true,metadata_thp:auto')
        elif use_default_allocator:
            pass
        else:
            find_tc = self.add_lib_preload(lib_type='tcmalloc')
            if find_tc:
                logger.info('Use TCMalloc memory allocator')
                return
            find_je = self.add_lib_preload(lib_type='jemalloc')
            if find_je:
                logger.info('Use JeMalloc memory allocator')
                return
            logger.warning('Neither TCMalloc nor JeMalloc is found in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib\n                            or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or\n                           %s/.local/lib/ so the LD_PRELOAD environment variable will not be set.\n                           This may drop the performance', expanduser('~'))

    def log_env_var(self, env_var_name=''):
        if env_var_name in os.environ:
            logger.info('%s=%s', env_var_name, os.environ[env_var_name])

    def set_env(self, env_name, env_value):
        if not env_value:
            logger.warning('%s is None', env_name)
        if env_name not in os.environ:
            os.environ[env_name] = env_value
        elif os.environ[env_name] != env_value:
            logger.warning('Overriding value with the one set in environment variable: %s. Value applied: %s. Value ignored: %s', env_name, os.environ[env_name], env_value)
        self.log_env_var(env_name)

    def set_multi_thread_and_allocator(self, ncores_per_instance, disable_iomp=False, set_kmp_affinity=True, enable_tcmalloc=True, enable_jemalloc=False, use_default_allocator=False):
        """
        Set multi-thread configuration and enable Intel openMP and TCMalloc/JeMalloc.

        By default, GNU openMP and PTMalloc are used in PyTorch. but Intel openMP and TCMalloc/JeMalloc are better alternatives
        to get performance benefit.
        """
        self.set_memory_allocator(enable_tcmalloc, enable_jemalloc, use_default_allocator)
        self.set_env('OMP_NUM_THREADS', str(ncores_per_instance))
        if not disable_iomp:
            find_iomp = self.add_lib_preload(lib_type='iomp5')
            if not find_iomp:
                msg = f'{self.msg_lib_notfound} you can use "conda install mkl" to install {{0}}'
                logger.warning(msg.format('iomp', 'iomp5'))
            else:
                logger.info('Using Intel OpenMP')
                if set_kmp_affinity:
                    self.set_env('KMP_AFFINITY', 'granularity=fine,compact,1,0')
                self.set_env('KMP_BLOCKTIME', '1')
        self.log_env_var('LD_PRELOAD')
    '\n     Launcher for single instance and multi-instance\n     '

    def launch(self, args):
        cores = []
        set_kmp_affinity = True
        enable_taskset = False
        if args.core_list:
            cores = [int(x) for x in args.core_list.split(',')]
            if args.ncores_per_instance == -1:
                raise RuntimeError('please specify the "--ncores-per-instance" if you have pass the --core-list params')
            elif args.ninstances > 1 and args.ncores_per_instance * args.ninstances < len(cores):
                logger.warning('only first %s cores will be used, but you specify %s cores in core_list', args.ncores_per_instance * args.ninstances, len(cores))
            else:
                args.ninstances = len(cores) // args.ncores_per_instance
        else:
            if args.use_logical_core:
                if args.node_id != -1:
                    cores = self.cpuinfo.get_node_logical_cores(args.node_id)
                else:
                    cores = self.cpuinfo.get_all_logical_cores()
                    set_kmp_affinity = False
            elif args.node_id != -1:
                cores = self.cpuinfo.get_node_physical_cores(args.node_id)
            else:
                cores = self.cpuinfo.get_all_physical_cores()
            if not args.multi_instance and args.ninstances == -1 and (args.ncores_per_instance == -1):
                args.ninstances = 1
                args.ncores_per_instance = len(cores)
            elif args.multi_instance and args.ninstances == -1 and (args.ncores_per_instance == -1):
                args.throughput_mode = True
            elif args.ncores_per_instance == -1 and args.ninstances != -1:
                if args.ninstances > len(cores):
                    raise RuntimeError(f'there are {len(cores)} total cores but you specify {args.ninstances} ninstances; please make sure ninstances <= total_cores)')
                else:
                    args.ncores_per_instance = len(cores) // args.ninstances
            elif args.ncores_per_instance != -1 and args.ninstances == -1:
                if not args.skip_cross_node_cores:
                    args.ninstances = len(cores) // args.ncores_per_instance
                else:
                    ncore_per_node = len(self.cpuinfo.node_physical_cores[0])
                    num_leftover_cores = ncore_per_node % args.ncores_per_instance
                    if args.ncores_per_instance > ncore_per_node:
                        logger.warning('there are %s core(s) per socket, but you specify %s ncores_per_instance and skip_cross_node_cores. Please make sure --ncores-per-instance < core(s) per socket', ncore_per_node, args.ncores_per_instance)
                        sys.exit(-1)
                    elif num_leftover_cores == 0:
                        logger.info('--skip-cross-node-cores is set, but there are no cross-node cores.')
                        args.ninstances = len(cores) // args.ncores_per_instance
                    else:
                        if args.ninstances != -1:
                            logger.warning("--skip-cross-node-cores is exclusive to --ninstances. --ninstances won't take effect even if it is set explicitly.")
                        i = 1
                        leftover_cores = set()
                        while ncore_per_node * i <= len(cores):
                            leftover_cores.update(cores[ncore_per_node * i - num_leftover_cores:ncore_per_node * i])
                            i += 1
                        cores = list(set(cores) - leftover_cores)
                        assert len(cores) % args.ncores_per_instance == 0
                        args.ninstances = len(cores) // args.ncores_per_instance
            elif args.ninstances * args.ncores_per_instance > len(cores):
                raise RuntimeError('Please make sure ninstances * ncores_per_instance <= total_cores')
            if args.latency_mode:
                logger.warning("--latency-mode is exclusive to --ninstances, --ncores-per-instance, --node-id and --use-logical-core. They won't take effect even they are set explicitly.")
                args.ncores_per_instance = 4
                cores = self.cpuinfo.get_all_physical_cores()
                args.ninstances = len(cores) // args.ncores_per_instance
            if args.throughput_mode:
                logger.warning("--throughput-mode is exclusive to --ninstances, --ncores-per-instance, --node-id and --use-logical-core. They won't take effect even they are set explicitly.")
                args.ninstances = self.cpuinfo.node_nums
                cores = self.cpuinfo.get_all_physical_cores()
                args.ncores_per_instance = len(cores) // args.ninstances
        if args.ninstances > 1 and args.rank != -1:
            logger.info('assigning %s cores for instance %s', args.ncores_per_instance, args.rank)
        if not args.disable_numactl:
            numactl_available = self.is_numactl_available()
            if not numactl_available:
                if not args.disable_taskset:
                    logger.warning('Core binding with numactl is not available. Disabling numactl and using taskset instead.                     This may affect performance in multi-socket system; please use numactl if memory binding is needed.')
                    args.disable_numactl = True
                    enable_taskset = True
                else:
                    logger.warning('Core binding with numactl is not available, and --disable_taskset is set.                     Please unset --disable_taskset to use taskset instead of numactl.')
                    sys.exit(-1)
        if not args.disable_taskset:
            enable_taskset = True
        self.set_multi_thread_and_allocator(args.ncores_per_instance, args.disable_iomp, set_kmp_affinity, args.enable_tcmalloc, args.enable_jemalloc, args.use_default_allocator)
        entrypoint = ''
        launch_args = {}
        launch_envs: Dict[int, Dict] = {}
        launch_tee = {}
        for i in range(args.ninstances):
            cmd = []
            cur_process_cores = ''
            if not args.disable_numactl or enable_taskset:
                if not args.disable_numactl:
                    cmd = ['numactl']
                elif enable_taskset:
                    cmd = ['taskset']
                cores = sorted(cores)
                if args.rank == -1:
                    core_list = cores[i * args.ncores_per_instance:(i + 1) * args.ncores_per_instance]
                else:
                    core_list = cores[args.rank * args.ncores_per_instance:(args.rank + 1) * args.ncores_per_instance]
                core_ranges: List[Dict] = []
                for core in core_list:
                    if len(core_ranges) == 0:
                        range_elem = {'start': core, 'end': core}
                        core_ranges.append(range_elem)
                    elif core - core_ranges[-1]['end'] == 1:
                        core_ranges[-1]['end'] = core
                    else:
                        range_elem = {'start': core, 'end': core}
                        core_ranges.append(range_elem)
                for r in core_ranges:
                    cur_process_cores = f'{cur_process_cores}{r['start']}-{r['end']},'
                cur_process_cores = cur_process_cores[:-1]
                if not args.disable_numactl:
                    numa_params = f'-C {cur_process_cores} '
                    numa_ids = ','.join([str(numa_id) for numa_id in self.cpuinfo.numa_aware_check(core_list)])
                    numa_params += f'-m {numa_ids}'
                    cmd.extend(numa_params.split())
                elif enable_taskset:
                    taskset_params = f'-c {cur_process_cores} '
                    cmd.extend(taskset_params.split())
            with_python = not args.no_python
            if with_python:
                cmd.append(sys.executable)
                cmd.append('-u')
            if args.module:
                cmd.append('-m')
            cmd.append(args.program)
            cmd.extend(args.program_args)
            cmd_s = ' '.join(cmd)
            logger.info(cmd_s)
            if entrypoint == '':
                entrypoint = cmd[0]
            del cmd[0]
            launch_args[i] = tuple(cmd)
            launch_envs[i] = {}
            launch_tee[i] = Std.ALL
            if args.rank != -1:
                break
        ctx = start_processes(name=args.log_file_prefix, entrypoint=entrypoint, args=launch_args, envs=launch_envs, log_dir=args.log_path, tee=launch_tee)
        ctx.wait()