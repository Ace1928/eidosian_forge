import os
import sys
import _yappi
import pickle
import threading
import warnings
import types
import inspect
import itertools
from contextlib import contextmanager
def _profile_dec(func):

    def wrapper(*args, **kwargs):
        if func._rec_level == 0:
            clear_stats()
            set_clock_type(clock_type)
            start(profile_builtins, profile_threads=False)
        func._rec_level += 1
        try:
            return func(*args, **kwargs)
        finally:
            func._rec_level -= 1
            if func._rec_level == 0:
                try:
                    stop()
                    if return_callback is None:
                        sys.stdout.write(LINESEP)
                        sys.stdout.write('Executed in {} {} clock seconds'.format(_fft(get_thread_stats()[0].ttot), clock_type.upper()))
                        sys.stdout.write(LINESEP)
                        get_func_stats().print_all()
                    else:
                        return_callback(func, get_func_stats())
                finally:
                    clear_stats()
    func._rec_level = 0
    return wrapper