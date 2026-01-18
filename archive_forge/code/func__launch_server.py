import datetime
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from typing import TYPE_CHECKING, Any, Dict, Optional
from wandb import _sentry, termlog
from wandb.env import error_reporting_enabled
from wandb.errors import Error
from wandb.sdk.lib.wburls import wburls
from wandb.util import get_core_path, get_module
from . import _startup_debug, port_file
from .service_base import ServiceInterface
from .service_sock import ServiceSockInterface
def _launch_server(self) -> None:
    """Launch server and set ports."""
    self._startup_debug_print('launch')
    kwargs: Dict[str, Any] = dict(close_fds=True)
    if platform.system() == 'Windows':
        kwargs.update(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        kwargs.update(start_new_session=True)
    pid = str(os.getpid())
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, f'port-{pid}.txt')
        executable = self._settings._executable
        exec_cmd_list = [executable, '-m']
        if os.environ.get('YEA_RUN_COVERAGE') and os.environ.get('COVERAGE_RCFILE'):
            exec_cmd_list += ['coverage', 'run', '-m']
        service_args = []
        core_path = get_core_path()
        if core_path:
            service_args.extend([core_path])
            if not error_reporting_enabled():
                service_args.append('--no-observability')
            if os.environ.get('WANDB_CORE_DEBUG', False):
                service_args.append('--debug')
            trace_filename = os.environ.get('_WANDB_TRACE')
            if trace_filename is not None:
                service_args.extend(['--trace', trace_filename])
            exec_cmd_list = []
            wandb_core = get_module('wandb_core')
            termlog(f'Using wandb-core version {wandb_core.__version__} as the SDK backend. Please refer to {wburls.get('wandb_core')} for more information.', repeat=False)
        else:
            service_args.extend(['wandb', 'service', '--debug'])
        service_args += ['--port-filename', fname, '--pid', pid]
        service_args.append('--serve-sock')
        if os.environ.get('WANDB_SERVICE_PROFILE') == 'memray':
            _ = get_module('memray', required='wandb service memory profiling requires memray, install with `pip install memray`')
            time_tag = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            output_file = f'wandb_service.memray.{time_tag}.bin'
            cli_executable = pathlib.Path(__file__).parent.parent.parent.parent / 'tools' / 'cli.py'
            exec_cmd_list = [executable, '-m', 'memray', 'run', '-o', output_file]
            service_args[0] = str(cli_executable)
            termlog(f'wandb service memory profiling enabled, output file: {output_file}')
            termlog(f'Convert to flamegraph with: `python -m memray flamegraph {output_file}`')
        try:
            internal_proc = subprocess.Popen(exec_cmd_list + service_args, env=os.environ, **kwargs)
        except Exception as e:
            _sentry.reraise(e)
        self._startup_debug_print('wait_ports')
        try:
            self._wait_for_ports(fname, proc=internal_proc)
        except Exception as e:
            _sentry.reraise(e)
        self._startup_debug_print('wait_ports_done')
        self._internal_proc = internal_proc
    self._startup_debug_print('launch_done')