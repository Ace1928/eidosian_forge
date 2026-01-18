import asyncio
import shutil
import subprocess
import sys
from pathlib import Path
import logging
class CpuProfilingManager:

    def __init__(self, profile_dir_path: str):
        self.profile_dir_path = Path(profile_dir_path)
        self.profile_dir_path.mkdir(exist_ok=True)

    async def trace_dump(self, pid: int, native: bool=False) -> (bool, str):
        pyspy = shutil.which('py-spy')
        if pyspy is None:
            return (False, 'py-spy is not installed')
        cmd = [pyspy, 'dump', '-p', str(pid)]
        if sys.platform == 'linux' and native:
            cmd.append('--native')
        if await _can_passwordless_sudo():
            cmd = ['sudo', '-n'] + cmd
        process = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            return (False, _format_failed_pyspy_command(cmd, stdout, stderr))
        else:
            return (True, stdout.decode('utf-8'))

    async def cpu_profile(self, pid: int, format='flamegraph', duration: float=5, native: bool=False) -> (bool, str):
        pyspy = shutil.which('py-spy')
        if pyspy is None:
            return (False, 'py-spy is not installed')
        if format not in ('flamegraph', 'raw', 'speedscope'):
            return (False, f'Invalid format {format}, ' + 'must be [flamegraph, raw, speedscope]')
        if format == 'flamegraph':
            extension = 'svg'
        else:
            extension = 'txt'
        profile_file_path = self.profile_dir_path / f'{format}_{pid}_cpu_profiling.{extension}'
        cmd = [pyspy, 'record', '-o', profile_file_path, '-p', str(pid), '-d', str(duration), '-f', format]
        if sys.platform == 'linux' and native:
            cmd.append('--native')
        if await _can_passwordless_sudo():
            cmd = ['sudo', '-n'] + cmd
        process = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            return (False, _format_failed_pyspy_command(cmd, stdout, stderr))
        else:
            return (True, open(profile_file_path, 'rb').read())