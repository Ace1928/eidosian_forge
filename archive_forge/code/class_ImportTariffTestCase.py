import os
from testtools import content
from .. import plugins as _mod_plugins
from .. import trace
from ..bzr.smart import medium
from ..controldir import ControlDir
from ..transport import remote
from . import TestCaseWithTransport
class ImportTariffTestCase(TestCaseWithTransport):
    """Check how many modules are loaded for some representative scenarios.

    See the Testing Guide in the developer documentation for more explanation.


    We must respect the setup used by the selftest command regarding
    plugins. This allows the user to control which plugins are in effect while
    running these tests and respect the import policies defined here.

    When failures are encountered for a given plugin, they can generally be
    addressed by using lazy import or lazy hook registration.
    """

    def setUp(self):
        self.preserved_env_vars = {}
        for name in ('BRZ_PLUGIN_PATH', 'BRZ_DISABLE_PLUGINS', 'BRZ_PLUGINS_AT'):
            self.preserved_env_vars[name] = os.environ.get(name)
        super().setUp()

    def start_brz_subprocess_with_import_check(self, args, stderr_file=None):
        """Run a bzr process and capture the imports.

        This is fairly expensive because we start a subprocess, so we aim to
        cover representative rather than exhaustive cases.
        """
        env_changes = dict(PYTHONVERBOSE='1', **self.preserved_env_vars)
        trace.mutter('Setting env for bzr subprocess: %r', env_changes)
        kwargs = dict(env_changes=env_changes, allow_plugins=False)
        if stderr_file:
            kwargs['stderr'] = stderr_file
        return self.start_brz_subprocess(args, **kwargs)

    def check_forbidden_modules(self, err, forbidden_imports):
        """Check for forbidden modules in stderr.

        :param err: Standard error
        :param forbidden_imports: List of forbidden modules
        """
        err = err.decode('utf-8')
        self.addDetail('subprocess_stderr', content.text_content(err))
        bad_modules = []
        for module_name in forbidden_imports:
            if err.find("\nimport '%s' " % module_name) != -1:
                bad_modules.append(module_name)
        if bad_modules:
            self.fail('command loaded forbidden modules %r' % (bad_modules,))

    def finish_brz_subprocess_with_import_check(self, process, args, forbidden_imports):
        """Finish subprocess and check specific modules have not been
        imported.

        :param forbidden_imports: List of fully-qualified Python module names
            that should not be loaded while running this command.
        """
        out, err = self.finish_brz_subprocess(process, universal_newlines=False, process_args=args)
        self.check_forbidden_modules(err, forbidden_imports)
        return (out, err)

    def run_command_check_imports(self, args, forbidden_imports):
        """Run bzr ARGS in a subprocess and check its imports.

        This is fairly expensive because we start a subprocess, so we aim to
        cover representative rather than exhaustive cases.

        :param forbidden_imports: List of fully-qualified Python module names
            that should not be loaded while running this command.
        """
        process = self.start_brz_subprocess_with_import_check(args)
        self.finish_brz_subprocess_with_import_check(process, args, forbidden_imports)