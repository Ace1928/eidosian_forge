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
class AnsibleVMwareTurboMode:

    def __init__(self):
        self.sessions = collections.defaultdict(dict)
        self.socket_path = None
        self.ttl = None
        self.debug_mode = None
        self.jobs_ongoing = {}

    async def ghost_killer(self):
        while True:
            await asyncio.sleep(self.ttl)
            running_jobs = {job_id: start_date for job_id, start_date in self.jobs_ongoing.items() if (datetime.now() - start_date).total_seconds() < 3600}
            if running_jobs:
                continue
            self.stop()

    async def handle(self, reader, writer):
        result = None
        self._watcher.cancel()
        self._watcher = self.loop.create_task(self.ghost_killer())
        job_id = str(uuid.uuid4())
        self.jobs_ongoing[job_id] = datetime.now()
        raw_data = await reader.read()
        if not raw_data:
            return
        plugin_type, content = pickle.loads(raw_data)

        def _terminate(result):
            writer.write(json.dumps(result).encode())
            writer.close()
        if plugin_type == 'module':
            result = await run_as_module(content, debug_mode=self.debug_mode)
        elif plugin_type == 'lookup':
            result = await run_as_lookup_plugin(content)
        _terminate(result)
        del self.jobs_ongoing[job_id]

    def handle_exception(self, loop, context):
        e = context.get('exception')
        traceback.print_exception(type(e), e, e.__traceback__)
        self.stop()

    def start(self):
        self.loop = asyncio.get_event_loop()
        self.loop.add_signal_handler(signal.SIGTERM, self.stop)
        self.loop.set_exception_handler(self.handle_exception)
        self._watcher = self.loop.create_task(self.ghost_killer())
        import sys
        try:
            from ansible.plugins.loader import init_plugin_loader
            init_plugin_loader()
        except ImportError:
            pass
        if sys.hexversion >= 50987185:
            self.loop.create_task(asyncio.start_unix_server(self.handle, path=self.socket_path))
        else:
            self.loop.create_task(asyncio.start_unix_server(self.handle, path=self.socket_path, loop=self.loop))
        self.loop.run_forever()

    def stop(self):
        os.unlink(self.socket_path)
        self.loop.stop()