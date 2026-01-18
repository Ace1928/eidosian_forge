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
def _add_multi_instance_params(parser):
    group = parser.add_argument_group('Multi-instance Parameters')
    group.add_argument('--ncores-per-instance', '--ncores_per_instance', metavar='\x08', default=-1, type=int, help='Cores per instance')
    group.add_argument('--ninstances', metavar='\x08', default=-1, type=int, help='For multi-instance, you should give the cores number you used for per instance.')
    group.add_argument('--skip-cross-node-cores', '--skip_cross_node_cores', action='store_true', default=False, help='If specified --ncores-per-instance, skips cross-node cores.')
    group.add_argument('--rank', metavar='\x08', default='-1', type=int, help='Specify instance index to assign ncores_per_instance for rank; otherwise ncores_per_instance will be assigned sequentially to ninstances. Please refer to https://github.com/intel/intel-extension-for-pytorch/blob/master/docs/tutorials/performance_tuning/launch_script.md')
    group.add_argument('--latency-mode', '--latency_mode', action='store_true', default=False, help='By default 4 core per instance and use all physical cores')
    group.add_argument('--throughput-mode', '--throughput_mode', action='store_true', default=False, help='By default one instance per node and use all physical cores')
    group.add_argument('--node-id', '--node_id', metavar='\x08', default=-1, type=int, help='node id for multi-instance, by default all nodes will be used')
    group.add_argument('--use-logical-core', '--use_logical_core', action='store_true', default=False, help='Whether only use physical cores')
    group.add_argument('--disable-numactl', '--disable_numactl', action='store_true', default=False, help='Disable numactl')
    group.add_argument('--disable-taskset', '--disable_taskset', action='store_true', default=False, help='Disable taskset')
    group.add_argument('--core-list', '--core_list', metavar='\x08', default=None, type=str, help='Specify the core list as "core_id, core_id, ....", otherwise, all the cores will be used.')
    group.add_argument('--log-path', '--log_path', metavar='\x08', default='', type=str, help='The log file directory. Default path is , which means disable logging to files.')
    group.add_argument('--log-file-prefix', '--log_file_prefix', metavar='\x08', default='run', type=str, help='log file prefix')