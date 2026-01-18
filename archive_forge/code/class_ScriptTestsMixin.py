from os import chdir, devnull, getcwd
from subprocess import PIPE, Popen
from sys import executable
from twisted.python.filepath import FilePath
from twisted.python.modules import getModule
from twisted.python.test.test_shellcomp import ZshScriptTestMixin
from twisted.trial.unittest import SkipTest, TestCase
class ScriptTestsMixin:
    """
    Mixin for L{TestCase} subclasses which defines a helper function for testing
    a Twisted-using script.
    """
    bin = getModule('twisted').pathEntry.filePath.child('bin')

    def scriptTest(self, name):
        """
        Verify that the given script runs and uses the version of Twisted
        currently being tested.

        This only works when running tests against a vcs checkout of Twisted,
        since it relies on the scripts being in the place they are kept in
        version control, and exercises their logic for finding the right version
        of Twisted to use in that situation.

        @param name: A path fragment, relative to the I{bin} directory of a
            Twisted source checkout, identifying a script to test.
        @type name: C{str}

        @raise SkipTest: if the script is not where it is expected to be.
        """
        script = self.bin.preauthChild(name)
        if not script.exists():
            raise SkipTest('Script tests do not apply to installed configuration.')
        from twisted.copyright import version
        scriptVersion = outputFromPythonScript(script, '--version')
        self.assertIn(str(version), scriptVersion)