import json
import os
import sys
import tempfile
from contextlib import contextmanager
from os.path import abspath
from os.path import join as pjoin
from subprocess import STDOUT, check_call, check_output
from ._in_process import _in_proc_script_path
class BuildBackendHookCaller:
    """A wrapper to call the build backend hooks for a source directory.
    """

    def __init__(self, source_dir, build_backend, backend_path=None, runner=None, python_executable=None):
        """
        :param source_dir: The source directory to invoke the build backend for
        :param build_backend: The build backend spec
        :param backend_path: Additional path entries for the build backend spec
        :param runner: The :ref:`subprocess runner <Subprocess Runners>` to use
        :param python_executable:
            The Python executable used to invoke the build backend
        """
        if runner is None:
            runner = default_subprocess_runner
        self.source_dir = abspath(source_dir)
        self.build_backend = build_backend
        if backend_path:
            backend_path = [norm_and_check(self.source_dir, p) for p in backend_path]
        self.backend_path = backend_path
        self._subprocess_runner = runner
        if not python_executable:
            python_executable = sys.executable
        self.python_executable = python_executable

    @contextmanager
    def subprocess_runner(self, runner):
        """A context manager for temporarily overriding the default
        :ref:`subprocess runner <Subprocess Runners>`.

        .. code-block:: python

            hook_caller = BuildBackendHookCaller(...)
            with hook_caller.subprocess_runner(quiet_subprocess_runner):
                ...
        """
        prev = self._subprocess_runner
        self._subprocess_runner = runner
        try:
            yield
        finally:
            self._subprocess_runner = prev

    def _supported_features(self):
        """Return the list of optional features supported by the backend."""
        return self._call_hook('_supported_features', {})

    def get_requires_for_build_wheel(self, config_settings=None):
        """Get additional dependencies required for building a wheel.

        :returns: A list of :pep:`dependency specifiers <508>`.
        :rtype: list[str]

        .. admonition:: Fallback

            If the build backend does not defined a hook with this name, an
            empty list will be returned.
        """
        return self._call_hook('get_requires_for_build_wheel', {'config_settings': config_settings})

    def prepare_metadata_for_build_wheel(self, metadata_directory, config_settings=None, _allow_fallback=True):
        """Prepare a ``*.dist-info`` folder with metadata for this project.

        :returns: Name of the newly created subfolder within
                  ``metadata_directory``, containing the metadata.
        :rtype: str

        .. admonition:: Fallback

            If the build backend does not define a hook with this name and
            ``_allow_fallback`` is truthy, the backend will be asked to build a
            wheel via the ``build_wheel`` hook and the dist-info extracted from
            that will be returned.
        """
        return self._call_hook('prepare_metadata_for_build_wheel', {'metadata_directory': abspath(metadata_directory), 'config_settings': config_settings, '_allow_fallback': _allow_fallback})

    def build_wheel(self, wheel_directory, config_settings=None, metadata_directory=None):
        """Build a wheel from this project.

        :returns:
            The name of the newly created wheel within ``wheel_directory``.

        .. admonition:: Interaction with fallback

            If the ``build_wheel`` hook was called in the fallback for
            :meth:`prepare_metadata_for_build_wheel`, the build backend would
            not be invoked. Instead, the previously built wheel will be copied
            to ``wheel_directory`` and the name of that file will be returned.
        """
        if metadata_directory is not None:
            metadata_directory = abspath(metadata_directory)
        return self._call_hook('build_wheel', {'wheel_directory': abspath(wheel_directory), 'config_settings': config_settings, 'metadata_directory': metadata_directory})

    def get_requires_for_build_editable(self, config_settings=None):
        """Get additional dependencies required for building an editable wheel.

        :returns: A list of :pep:`dependency specifiers <508>`.
        :rtype: list[str]

        .. admonition:: Fallback

            If the build backend does not defined a hook with this name, an
            empty list will be returned.
        """
        return self._call_hook('get_requires_for_build_editable', {'config_settings': config_settings})

    def prepare_metadata_for_build_editable(self, metadata_directory, config_settings=None, _allow_fallback=True):
        """Prepare a ``*.dist-info`` folder with metadata for this project.

        :returns: Name of the newly created subfolder within
                  ``metadata_directory``, containing the metadata.
        :rtype: str

        .. admonition:: Fallback

            If the build backend does not define a hook with this name and
            ``_allow_fallback`` is truthy, the backend will be asked to build a
            wheel via the ``build_editable`` hook and the dist-info
            extracted from that will be returned.
        """
        return self._call_hook('prepare_metadata_for_build_editable', {'metadata_directory': abspath(metadata_directory), 'config_settings': config_settings, '_allow_fallback': _allow_fallback})

    def build_editable(self, wheel_directory, config_settings=None, metadata_directory=None):
        """Build an editable wheel from this project.

        :returns:
            The name of the newly created wheel within ``wheel_directory``.

        .. admonition:: Interaction with fallback

            If the ``build_editable`` hook was called in the fallback for
            :meth:`prepare_metadata_for_build_editable`, the build backend
            would not be invoked. Instead, the previously built wheel will be
            copied to ``wheel_directory`` and the name of that file will be
            returned.
        """
        if metadata_directory is not None:
            metadata_directory = abspath(metadata_directory)
        return self._call_hook('build_editable', {'wheel_directory': abspath(wheel_directory), 'config_settings': config_settings, 'metadata_directory': metadata_directory})

    def get_requires_for_build_sdist(self, config_settings=None):
        """Get additional dependencies required for building an sdist.

        :returns: A list of :pep:`dependency specifiers <508>`.
        :rtype: list[str]
        """
        return self._call_hook('get_requires_for_build_sdist', {'config_settings': config_settings})

    def build_sdist(self, sdist_directory, config_settings=None):
        """Build an sdist from this project.

        :returns:
            The name of the newly created sdist within ``wheel_directory``.
        """
        return self._call_hook('build_sdist', {'sdist_directory': abspath(sdist_directory), 'config_settings': config_settings})

    def _call_hook(self, hook_name, kwargs):
        extra_environ = {'PEP517_BUILD_BACKEND': self.build_backend}
        if self.backend_path:
            backend_path = os.pathsep.join(self.backend_path)
            extra_environ['PEP517_BACKEND_PATH'] = backend_path
        with tempfile.TemporaryDirectory() as td:
            hook_input = {'kwargs': kwargs}
            write_json(hook_input, pjoin(td, 'input.json'), indent=2)
            with _in_proc_script_path() as script:
                python = self.python_executable
                self._subprocess_runner([python, abspath(str(script)), hook_name, td], cwd=self.source_dir, extra_environ=extra_environ)
            data = read_json(pjoin(td, 'output.json'))
            if data.get('unsupported'):
                raise UnsupportedOperation(data.get('traceback', ''))
            if data.get('no_backend'):
                raise BackendUnavailable(data.get('traceback', ''))
            if data.get('backend_invalid'):
                raise BackendInvalid(backend_name=self.build_backend, backend_path=self.backend_path, message=data.get('backend_error', ''))
            if data.get('hook_missing'):
                raise HookMissing(data.get('missing_hook_name') or hook_name)
            return data['return_val']