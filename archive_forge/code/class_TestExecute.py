import asyncio
import concurrent.futures
import copy
import datetime
import functools
import os
import re
import sys
import threading
import warnings
from base64 import b64decode, b64encode
from queue import Empty
from typing import Any
from unittest.mock import MagicMock, Mock
import nbformat
import pytest
import xmltodict
from flaky import flaky  # type:ignore
from jupyter_client import KernelClient, KernelManager
from jupyter_client._version import version_info
from jupyter_client.kernelspec import KernelSpecManager
from nbconvert.filters import strip_ansi
from nbformat import NotebookNode
from testpath import modified_env
from traitlets import TraitError
from nbclient import NotebookClient, execute
from nbclient.exceptions import CellExecutionError
from .base import NBClientTestsBase
class TestExecute(NBClientTestsBase):
    """Contains test functions for execute.py"""
    maxDiff = None

    def test_constructor(self):
        NotebookClient(nbformat.v4.new_notebook())

    def test_populate_language_info(self):
        nb = nbformat.v4.new_notebook()
        executor = NotebookClient(nb, kernel_name='python')
        nb = executor.execute()
        assert 'language_info' in nb.metadata

    def test_empty_path(self):
        """Can the kernel be started when the path is empty?"""
        filename = os.path.join(current_dir, 'files', 'HelloWorld.ipynb')
        res = self.build_resources()
        res['metadata']['path'] = ''
        input_nb, output_nb = run_notebook(filename, {}, res)
        assert_notebooks_equal(input_nb, output_nb)

    @pytest.mark.xfail('python3' not in KernelSpecManager().find_kernel_specs(), reason='requires a python3 kernelspec')
    def test_empty_kernel_name(self):
        """Can kernel in nb metadata be found when an empty string is passed?

        Note: this pattern should be discouraged in practice.
        Passing in no kernel_name to NotebookClient is recommended instead.
        """
        filename = os.path.join(current_dir, 'files', 'UnicodePy3.ipynb')
        res = self.build_resources()
        input_nb, output_nb = run_notebook(filename, {'kernel_name': ''}, res)
        assert_notebooks_equal(input_nb, output_nb)
        with pytest.raises(TraitError):
            input_nb, output_nb = run_notebook(filename, {'kernel_name': None}, res)

    def test_disable_stdin(self):
        """Test disabling standard input"""
        filename = os.path.join(current_dir, 'files', 'Disable Stdin.ipynb')
        res = self.build_resources()
        res['metadata']['path'] = os.path.dirname(filename)
        input_nb, output_nb = run_notebook(filename, {'allow_errors': True}, res)
        self.assertEqual(len(output_nb['cells']), 1)
        self.assertEqual(len(output_nb['cells'][0]['outputs']), 1)
        output = output_nb['cells'][0]['outputs'][0]
        self.assertEqual(output['output_type'], 'error')
        self.assertEqual(output['ename'], 'StdinNotImplementedError')
        self.assertEqual(output['evalue'], 'raw_input was called, but this frontend does not support input requests.')

    def test_timeout(self):
        """Check that an error is raised when a computation times out"""
        filename = os.path.join(current_dir, 'files', 'Interrupt.ipynb')
        res = self.build_resources()
        res['metadata']['path'] = os.path.dirname(filename)
        with pytest.raises(TimeoutError) as err:
            run_notebook(filename, {'timeout': 1}, res)
        self.assertEqual(str(err.value.args[0]), 'A cell timed out while it was being executed, after 1 seconds.\nThe message was: Cell execution timed out.\nHere is a preview of the cell contents:\n-------------------\nwhile True: continue\n-------------------\n')

    def test_timeout_func(self):
        """Check that an error is raised when a computation times out"""
        filename = os.path.join(current_dir, 'files', 'Interrupt.ipynb')
        res = self.build_resources()
        res['metadata']['path'] = os.path.dirname(filename)

        def timeout_func(source):
            return 10
        with pytest.raises(TimeoutError):
            run_notebook(filename, {'timeout_func': timeout_func}, res)

    def test_sync_kernel_manager(self):
        nb = nbformat.v4.new_notebook()
        executor = NotebookClient(nb, kernel_name='python', kernel_manager_class=KernelManager)
        nb = executor.execute()
        assert 'language_info' in nb.metadata
        with executor.setup_kernel():
            assert executor.kc is not None
            info_msg = executor.wait_for_reply(executor.kc.kernel_info())
            assert info_msg is not None
            assert 'name' in info_msg['content']['language_info']

    @flaky
    def test_kernel_death_after_timeout(self):
        """Check that an error is raised when the kernel is_alive is false after a cell timed out"""
        filename = os.path.join(current_dir, 'files', 'Interrupt.ipynb')
        with open(filename) as f:
            input_nb = nbformat.read(f, 4)
        res = self.build_resources()
        res['metadata']['path'] = os.path.dirname(filename)
        executor = NotebookClient(input_nb, timeout=1)
        with pytest.raises(TimeoutError):
            executor.execute()
        km = executor.create_kernel_manager()

        async def is_alive():
            return False
        km.is_alive = is_alive
        with pytest.raises((RuntimeError, TimeoutError)):
            input_nb, output_nb = executor.execute()

    def test_kernel_death_during_execution(self):
        """Check that an error is raised when the kernel is_alive is false during a cell
        execution.
        """
        filename = os.path.join(current_dir, 'files', 'Autokill.ipynb')
        with open(filename) as f:
            input_nb = nbformat.read(f, 4)
        executor = NotebookClient(input_nb)
        with pytest.raises(RuntimeError):
            executor.execute()

    def test_allow_errors(self):
        """
        Check that conversion halts if ``allow_errors`` is False.
        """
        filename = os.path.join(current_dir, 'files', 'Skip Exceptions.ipynb')
        res = self.build_resources()
        res['metadata']['path'] = os.path.dirname(filename)
        with pytest.raises(CellExecutionError) as exc:
            run_notebook(filename, {'allow_errors': False}, res)
        assert isinstance(str(exc.value), str)
        exc_str = strip_ansi(str(exc.value))
        if not sys.platform.startswith('win'):
            assert '# üñîçø∂é' in exc_str

    def test_force_raise_errors(self):
        """
        Check that conversion halts if the ``force_raise_errors`` traitlet on
        NotebookClient is set to True.
        """
        filename = os.path.join(current_dir, 'files', 'Skip Exceptions with Cell Tags.ipynb')
        res = self.build_resources()
        res['metadata']['path'] = os.path.dirname(filename)
        with pytest.raises(CellExecutionError) as exc:
            run_notebook(filename, {'force_raise_errors': True}, res)
        exc_str = strip_ansi(str(exc.value))
        assert 'Exception: message' in exc_str
        if not sys.platform.startswith('win'):
            assert '# üñîçø∂é' in exc_str
        assert 'stderr' in exc_str
        assert 'stdout' in exc_str
        assert 'hello\n' in exc_str
        assert 'errorred\n' in exc_str
        assert '\n'.join(['', '----- stdout -----', 'hello', '---']) in exc_str
        assert '\n'.join(['', '----- stderr -----', 'errorred', '---']) in exc_str

    def test_reset_kernel_client(self):
        filename = os.path.join(current_dir, 'files', 'HelloWorld.ipynb')
        with open(filename) as f:
            input_nb = nbformat.read(f, 4)
        executor = NotebookClient(input_nb, resources=self.build_resources())
        executor.execute(cleanup_kc=False)
        kc = executor.kc
        assert kc is not None
        executor.execute(cleanup_kc=False)
        assert kc == executor.kc
        executor.execute(reset_kc=True, cleanup_kc=False)
        assert kc != executor.kc

    def test_cleanup_kernel_client(self):
        filename = os.path.join(current_dir, 'files', 'HelloWorld.ipynb')
        with open(filename) as f:
            input_nb = nbformat.read(f, 4)
        executor = NotebookClient(input_nb, resources=self.build_resources())
        executor.execute()
        assert executor.kc is None
        executor.execute(cleanup_kc=False)
        assert executor.kc is not None

    def test_custom_kernel_manager(self):
        from .fake_kernelmanager import FakeCustomKernelManager
        filename = os.path.join(current_dir, 'files', 'HelloWorld.ipynb')
        with open(filename) as f:
            input_nb = nbformat.read(f, 4)
        cleaned_input_nb = copy.deepcopy(input_nb)
        for cell in cleaned_input_nb.cells:
            if 'execution_count' in cell:
                del cell['execution_count']
            cell['outputs'] = []
        executor = NotebookClient(cleaned_input_nb, resources=self.build_resources(), kernel_manager_class=FakeCustomKernelManager)
        with modified_env({'COLUMNS': '80', 'LINES': '24'}):
            executor.execute()
        expected = FakeCustomKernelManager.expected_methods.items()
        for method, call_count in expected:
            self.assertNotEqual(call_count, 0, f'{method} was called')

    def test_process_message_wrapper(self):
        outputs: list = []

        class WrappedPreProc(NotebookClient):

            def process_message(self, msg, cell, cell_index):
                result = super().process_message(msg, cell, cell_index)
                if result:
                    outputs.append(result)
                return result
        current_dir = os.path.dirname(__file__)
        filename = os.path.join(current_dir, 'files', 'HelloWorld.ipynb')
        with open(filename) as f:
            input_nb = nbformat.read(f, 4)
        original = copy.deepcopy(input_nb)
        wpp = WrappedPreProc(input_nb)
        executed = wpp.execute()
        assert outputs == [{'name': 'stdout', 'output_type': 'stream', 'text': 'Hello World\n'}]
        assert_notebooks_equal(original, executed)

    def test_execute_function(self):
        filename = os.path.join(current_dir, 'files', 'HelloWorld.ipynb')
        with open(filename) as f:
            input_nb = nbformat.read(f, 4)
        original = copy.deepcopy(input_nb)
        executed = execute(original, os.path.dirname(filename))
        assert_notebooks_equal(original, executed)

    def test_widgets(self):
        """Runs a test notebook with widgets and checks the widget state is saved."""
        input_file = os.path.join(current_dir, 'files', 'JupyterWidgets.ipynb')
        opts = {'kernel_name': 'python'}
        res = self.build_resources()
        res['metadata']['path'] = os.path.dirname(input_file)
        input_nb, output_nb = run_notebook(input_file, opts, res)
        output_data = [output.get('data', {}) for cell in output_nb['cells'] for output in cell['outputs']]
        model_ids = [data['application/vnd.jupyter.widget-view+json']['model_id'] for data in output_data if 'application/vnd.jupyter.widget-view+json' in data]
        wdata = output_nb['metadata']['widgets']['application/vnd.jupyter.widget-state+json']
        for k in model_ids:
            d = wdata['state'][k]
            assert 'model_name' in d
            assert 'model_module' in d
            assert 'state' in d
        assert 'version_major' in wdata
        assert 'version_minor' in wdata

    def test_execution_hook(self):
        filename = os.path.join(current_dir, 'files', 'HelloWorld.ipynb')
        with open(filename) as f:
            input_nb = nbformat.read(f, 4)
        executor, hooks = get_executor_with_hooks(nb=input_nb)
        executor.execute()
        hooks['on_cell_start'].assert_called_once()
        hooks['on_cell_execute'].assert_called_once()
        hooks['on_cell_complete'].assert_called_once()
        hooks['on_cell_executed'].assert_called_once()
        hooks['on_cell_error'].assert_not_called()
        hooks['on_notebook_start'].assert_called_once()
        hooks['on_notebook_complete'].assert_called_once()
        hooks['on_notebook_error'].assert_not_called()

    def test_error_execution_hook_error(self):
        filename = os.path.join(current_dir, 'files', 'Error.ipynb')
        with open(filename) as f:
            input_nb = nbformat.read(f, 4)
        executor, hooks = get_executor_with_hooks(nb=input_nb)
        with pytest.raises(CellExecutionError):
            executor.execute()
        hooks['on_cell_start'].assert_called_once()
        hooks['on_cell_execute'].assert_called_once()
        hooks['on_cell_complete'].assert_called_once()
        hooks['on_cell_executed'].assert_called_once()
        hooks['on_cell_error'].assert_called_once()
        hooks['on_notebook_start'].assert_called_once()
        hooks['on_notebook_complete'].assert_called_once()
        hooks['on_notebook_error'].assert_not_called()

    def test_error_notebook_hook(self):
        filename = os.path.join(current_dir, 'files', 'Autokill.ipynb')
        with open(filename) as f:
            input_nb = nbformat.read(f, 4)
        executor, hooks = get_executor_with_hooks(nb=input_nb)
        with pytest.raises(RuntimeError):
            executor.execute()
        hooks['on_cell_start'].assert_called_once()
        hooks['on_cell_execute'].assert_called_once()
        hooks['on_cell_complete'].assert_called_once()
        hooks['on_cell_executed'].assert_not_called()
        hooks['on_cell_error'].assert_not_called()
        hooks['on_notebook_start'].assert_called_once()
        hooks['on_notebook_complete'].assert_called_once()
        hooks['on_notebook_error'].assert_called_once()

    def test_async_execution_hook(self):
        filename = os.path.join(current_dir, 'files', 'HelloWorld.ipynb')
        with open(filename) as f:
            input_nb = nbformat.read(f, 4)
        executor, hooks = get_executor_with_hooks(nb=input_nb)
        executor.execute()
        hooks['on_cell_start'].assert_called_once()
        hooks['on_cell_execute'].assert_called_once()
        hooks['on_cell_complete'].assert_called_once()
        hooks['on_cell_executed'].assert_called_once()
        hooks['on_cell_error'].assert_not_called()
        hooks['on_notebook_start'].assert_called_once()
        hooks['on_notebook_complete'].assert_called_once()
        hooks['on_notebook_error'].assert_not_called()

    def test_error_async_execution_hook(self):
        filename = os.path.join(current_dir, 'files', 'Error.ipynb')
        with open(filename) as f:
            input_nb = nbformat.read(f, 4)
        executor, hooks = get_executor_with_hooks(nb=input_nb)
        with pytest.raises(CellExecutionError):
            executor.execute()
        hooks['on_cell_start'].assert_called_once()
        hooks['on_cell_execute'].assert_called_once()
        hooks['on_cell_complete'].assert_called_once()
        hooks['on_cell_executed'].assert_called_once()
        hooks['on_cell_error'].assert_called_once()
        hooks['on_notebook_start'].assert_called_once()
        hooks['on_notebook_complete'].assert_called_once()
        hooks['on_notebook_error'].assert_not_called()