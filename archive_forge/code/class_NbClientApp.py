import logging
import pathlib
import sys
from textwrap import dedent
import nbformat
from jupyter_core.application import JupyterApp
from traitlets import Bool, Integer, List, Unicode, default
from traitlets.config import catch_config_error
from nbclient import __version__
from .client import NotebookClient
class NbClientApp(JupyterApp):
    """
    An application used to execute notebook files (``*.ipynb``)
    """
    version = Unicode(__version__)
    name = 'jupyter-execute'
    aliases = nbclient_aliases
    flags = nbclient_flags
    description = 'An application used to execute notebook files (*.ipynb)'
    notebooks = List([], help='Path of notebooks to convert').tag(config=True)
    timeout: int = Integer(None, allow_none=True, help=dedent('\n            The time to wait (in seconds) for output from executions.\n            If a cell execution takes longer, a TimeoutError is raised.\n            ``-1`` will disable the timeout.\n            ')).tag(config=True)
    startup_timeout: int = Integer(60, help=dedent('\n            The time to wait (in seconds) for the kernel to start.\n            If kernel startup takes longer, a RuntimeError is\n            raised.\n            ')).tag(config=True)
    allow_errors: bool = Bool(False, help=dedent('\n            When a cell raises an error the default behavior is that\n            execution is stopped and a :py:class:`nbclient.exceptions.CellExecutionError`\n            is raised.\n            If this flag is provided, errors are ignored and execution\n            is continued until the end of the notebook.\n            ')).tag(config=True)
    skip_cells_with_tag: str = Unicode('skip-execution', help=dedent('\n            Name of the cell tag to use to denote a cell that should be skipped.\n            ')).tag(config=True)
    kernel_name: str = Unicode('', help=dedent('\n            Name of kernel to use to execute the cells.\n            If not set, use the kernel_spec embedded in the notebook.\n            ')).tag(config=True)

    @default('log_level')
    def _log_level_default(self):
        return logging.INFO

    @catch_config_error
    def initialize(self, argv=None):
        """Initialize the app."""
        super().initialize(argv)
        self.notebooks = self.get_notebooks()
        if not self.notebooks:
            sys.exit(-1)
        [self.run_notebook(path) for path in self.notebooks]

    def get_notebooks(self):
        """Get the notebooks for the app."""
        if self.extra_args:
            notebooks = self.extra_args
        else:
            notebooks = self.notebooks
        return notebooks

    def run_notebook(self, notebook_path):
        """Run a notebook by path."""
        self.log.info(f'Executing {notebook_path}')
        name = notebook_path.replace('.ipynb', '')
        path = pathlib.Path(notebook_path).parent.absolute()
        input_path = f'{name}.ipynb'
        with open(input_path) as f:
            nb = nbformat.read(f, as_version=4)
        client = NotebookClient(nb, timeout=self.timeout, startup_timeout=self.startup_timeout, skip_cells_with_tag=self.skip_cells_with_tag, allow_errors=self.allow_errors, kernel_name=self.kernel_name, resources={'metadata': {'path': path}})
        client.execute()