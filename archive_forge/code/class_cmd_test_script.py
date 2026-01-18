import os
from . import commands, option
class cmd_test_script(commands.Command):
    """Run a shell-like test from a file."""
    hidden = True
    takes_args = ['infile']
    takes_options = [option.Option('null-output', help='Null command outputs match any output.')]

    @commands.display_command
    def run(self, infile, null_output=False):
        from breezy import tests
        from breezy.tests.script import TestCaseWithTransportAndScript
        with open(infile) as f:
            script = f.read()

        class Test(TestCaseWithTransportAndScript):
            script = None

            def test_it(self):
                self.run_script(script, null_output_matches_anything=null_output)
        runner = tests.TextTestRunner(stream=self.outf)
        test = Test('test_it')
        test.path = os.path.realpath(infile)
        res = runner.run(test)
        return len(res.errors) + len(res.failures)