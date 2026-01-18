import multiprocessing
import os
import signal
import subprocess
import gyp.common
import gyp.xcode_emulation
def CallGenerateOutputForConfig(arglist):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    target_list, target_dicts, data, params, config_name = arglist
    GenerateOutputForConfig(target_list, target_dicts, data, params, config_name)