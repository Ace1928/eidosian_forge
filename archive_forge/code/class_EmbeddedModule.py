import argparse
import asyncio
from datetime import datetime
import importlib
import inspect  # pylint: disable=syntax-error
import io
import json
import collections  # pylint: disable=syntax-error
import os
import signal
import sys
import traceback
import zipfile
from zipimport import zipimporter
import pickle
import uuid
import ansible.module_utils.basic
class EmbeddedModule:

    def __init__(self, ansiblez_path, params):
        self.ansiblez_path = ansiblez_path
        self.collection_name, self.module_name = self.find_module_name()
        self.params = params
        self.module_class = None
        self.debug_mode = False
        self.module_path = 'ansible_collections.{collection_name}.plugins.modules.{module_name}'.format(collection_name=self.collection_name, module_name=self.module_name)

    def find_module_name(self):
        with zipfile.ZipFile(self.ansiblez_path) as zip:
            for path in zip.namelist():
                if not path.startswith('ansible_collections'):
                    continue
                if not path.endswith('.py'):
                    continue
                if path.endswith('__init__.py'):
                    continue
                splitted = path.split('/')
                if len(splitted) != 6:
                    continue
                if splitted[-3:-1] != ['plugins', 'modules']:
                    continue
                collection = '.'.join(splitted[1:3])
                name = splitted[-1][:-3]
                return (collection, name)
        raise Exception('Cannot find module name')

    async def load(self):
        async with sys_path_lock:
            sys.path.insert(0, self.ansiblez_path)
            for path, module in sorted(tuple(sys.modules.items())):
                if path and module and path.startswith('ansible_collections'):
                    try:
                        prefix = sys.modules[path].__loader__.prefix
                    except AttributeError:
                        continue
                    if hasattr(sys.modules[path], '__path__'):
                        py_path = self.ansiblez_path + os.sep + prefix
                        my_loader = zipimporter(py_path)
                        sys.modules[path].__loader__ = my_loader
                        try:
                            importlib.reload(sys.modules[path])
                        except ModuleNotFoundError:
                            pass
            self.module_class = importlib.import_module(self.module_path)

    async def unload(self):
        async with sys_path_lock:
            sys.path = [i for i in sys.path if i != self.ansiblez_path]

    def create_profiler(self):
        if self.debug_mode:
            import cProfile
            pr = cProfile.Profile()
            pr.enable()
            return pr

    def print_profiling_info(self, pr):
        if self.debug_mode:
            import pstats
            sortby = pstats.SortKey.CUMULATIVE
            ps = pstats.Stats(pr).sort_stats(sortby)
            ps.print_stats(20)

    def print_backtrace(self, backtrace):
        if self.debug_mode:
            print(backtrace)

    async def run(self):

        class FakeStdin:
            buffer = None
        from .exceptions import EmbeddedModuleFailure, EmbeddedModuleUnexpectedFailure, EmbeddedModuleSuccess
        _fake_stdin = FakeStdin()
        _fake_stdin.buffer = io.BytesIO(self.params.encode())
        sys.stdin = _fake_stdin
        sys.argv = sys.argv[:1]
        ansible.module_utils.basic._ANSIBLE_ARGS = None
        pr = self.create_profiler()
        if not hasattr(self.module_class, 'main'):
            raise EmbeddedModuleFailure('No main() found!')
        try:
            if inspect.iscoroutinefunction(self.module_class.main):
                await self.module_class.main()
            elif pr:
                pr.runcall(self.module_class.main)
            else:
                self.module_class.main()
        except EmbeddedModuleSuccess as e:
            self.print_profiling_info(pr)
            return e.kwargs
        except EmbeddedModuleFailure as e:
            backtrace = traceback.format_exc()
            self.print_backtrace(backtrace)
            raise
        except Exception as e:
            backtrace = traceback.format_exc()
            self.print_backtrace(backtrace)
            raise EmbeddedModuleUnexpectedFailure(str(backtrace))
        else:
            raise EmbeddedModuleUnexpectedFailure('Likely a bug: exit_json() or fail_json() should be called during the module excution')