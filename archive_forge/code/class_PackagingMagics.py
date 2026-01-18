import functools
import re
import shlex
import sys
from pathlib import Path
from IPython.core.magic import Magics, magics_class, line_magic
@magics_class
class PackagingMagics(Magics):
    """Magics related to packaging & installation"""

    @line_magic
    def pip(self, line):
        """Run the pip package manager within the current kernel.

        Usage:
          %pip install [pkgs]
        """
        python = sys.executable
        if sys.platform == 'win32':
            python = '"' + python + '"'
        else:
            python = shlex.quote(python)
        self.shell.system(' '.join([python, '-m', 'pip', line]))
        print('Note: you may need to restart the kernel to use updated packages.')

    def _run_command(self, cmd, line):
        args = shlex.split(line)
        command = args[0] if len(args) > 0 else ''
        args = args[1:] if len(args) > 1 else ['']
        extra_args = []
        stdin_disabled = getattr(self.shell, 'kernel', None) is not None
        needs_yes = command in CONDA_COMMANDS_REQUIRING_YES
        has_yes = set(args).intersection(CONDA_YES_FLAGS)
        if stdin_disabled and needs_yes and (not has_yes):
            extra_args.append('--yes')
        needs_prefix = command in CONDA_COMMANDS_REQUIRING_PREFIX
        has_prefix = set(args).intersection(CONDA_ENV_FLAGS)
        if needs_prefix and (not has_prefix):
            extra_args.extend(['--prefix', sys.prefix])
        self.shell.system(' '.join([cmd, command] + extra_args + args))
        print('\nNote: you may need to restart the kernel to use updated packages.')

    @line_magic
    @is_conda_environment
    def conda(self, line):
        """Run the conda package manager within the current kernel.

        Usage:
          %conda install [pkgs]
        """
        conda = _get_conda_like_executable('conda')
        self._run_command(conda, line)

    @line_magic
    @is_conda_environment
    def mamba(self, line):
        """Run the mamba package manager within the current kernel.

        Usage:
          %mamba install [pkgs]
        """
        mamba = _get_conda_like_executable('mamba')
        self._run_command(mamba, line)

    @line_magic
    @is_conda_environment
    def micromamba(self, line):
        """Run the conda package manager within the current kernel.

        Usage:
          %micromamba install [pkgs]
        """
        micromamba = _get_conda_like_executable('micromamba')
        self._run_command(micromamba, line)