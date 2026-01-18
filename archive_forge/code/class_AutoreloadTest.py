import os
import shutil
import subprocess
from subprocess import Popen
import sys
from tempfile import mkdtemp
import textwrap
import time
import unittest
import sys
import sys
import testapp
import os
import sys
import os
import sys
class AutoreloadTest(unittest.TestCase):

    def setUp(self):
        self.maxDiff = 1024
        self.path = mkdtemp()
        self.write_files({'run_twice_magic.py': '\n                    import os\n                    import sys\n\n                    import tornado.autoreload\n\n                    sys.stdout.flush()\n\n                    if "TESTAPP_STARTED" not in os.environ:\n                        os.environ["TESTAPP_STARTED"] = "1"\n                        tornado.autoreload._reload()\n                    else:\n                        os._exit(0)\n                '})

    def tearDown(self):
        try:
            shutil.rmtree(self.path)
        except OSError:
            time.sleep(1)
            shutil.rmtree(self.path)

    def write_files(self, tree, base_path=None):
        """Write a directory tree to self.path.

        tree is a dictionary mapping file names to contents, or
        sub-dictionaries representing subdirectories.
        """
        if base_path is None:
            base_path = self.path
        for name, contents in tree.items():
            if isinstance(contents, dict):
                os.mkdir(os.path.join(base_path, name))
                self.write_files(contents, os.path.join(base_path, name))
            else:
                with open(os.path.join(base_path, name), 'w', encoding='utf-8') as f:
                    f.write(textwrap.dedent(contents))

    def run_subprocess(self, args):
        pythonpath = os.getcwd()
        if 'PYTHONPATH' in os.environ:
            pythonpath += os.pathsep + os.environ['PYTHONPATH']
        p = Popen(args, stdout=subprocess.PIPE, env=dict(os.environ, PYTHONPATH=pythonpath), cwd=self.path, universal_newlines=True, encoding='utf-8')
        for i in range(40):
            if p.poll() is not None:
                break
            time.sleep(0.1)
        else:
            p.kill()
            raise Exception('subprocess failed to terminate')
        out = p.communicate()[0]
        self.assertEqual(p.returncode, 0)
        return out

    def test_reload(self):
        main = 'import sys\n\n# In module mode, the path is set to the parent directory and we can import testapp.\ntry:\n    import testapp\nexcept ImportError:\n    print("import testapp failed")\nelse:\n    print("import testapp succeeded")\n\nspec = getattr(sys.modules[__name__], \'__spec__\', None)\nprint(f"Starting {__name__=}, __spec__.name={getattr(spec, \'name\', None)}")\nexec(open("run_twice_magic.py").read())\n'
        self.write_files({'testapp': {'__init__.py': '', '__main__.py': main}})
        for wrapper in [False, True]:
            with self.subTest(wrapper=wrapper):
                with self.subTest(mode='module'):
                    if wrapper:
                        base_args = [sys.executable, '-m', 'tornado.autoreload']
                    else:
                        base_args = [sys.executable]
                    out = self.run_subprocess(base_args + ['-m', 'testapp'])
                    self.assertEqual(out, ('import testapp succeeded\n' + "Starting __name__='__main__', __spec__.name=testapp.__main__\n") * 2)
                with self.subTest(mode='file'):
                    out = self.run_subprocess(base_args + ['testapp/__main__.py'])
                    expect_import = 'import testapp succeeded' if wrapper else 'import testapp failed'
                    self.assertEqual(out, f"{expect_import}\nStarting __name__='__main__', __spec__.name=None\n" * 2)
                with self.subTest(mode='directory'):
                    out = self.run_subprocess(base_args + ['testapp'])
                    expect_import = 'import testapp succeeded' if wrapper else 'import testapp failed'
                    self.assertEqual(out, f"{expect_import}\nStarting __name__='__main__', __spec__.name=__main__\n" * 2)

    def test_reload_wrapper_preservation(self):
        main = 'import sys\n\n# This import will fail if path is not set up correctly\nimport testapp\n\nif \'tornado.autoreload\' not in sys.modules:\n    raise Exception(\'started without autoreload wrapper\')\n\nprint(\'Starting\')\nexec(open("run_twice_magic.py").read())\n'
        self.write_files({'testapp': {'__init__.py': '', '__main__.py': main}})
        out = self.run_subprocess([sys.executable, '-m', 'tornado.autoreload', '-m', 'testapp'])
        self.assertEqual(out, 'Starting\n' * 2)

    def test_reload_wrapper_args(self):
        main = 'import os\nimport sys\n\nprint(os.path.basename(sys.argv[0]))\nprint(f\'argv={sys.argv[1:]}\')\nexec(open("run_twice_magic.py").read())\n'
        self.write_files({'main.py': main})
        out = self.run_subprocess([sys.executable, '-m', 'tornado.autoreload', 'main.py', 'arg1', '--arg2', '-m', 'arg3'])
        self.assertEqual(out, "main.py\nargv=['arg1', '--arg2', '-m', 'arg3']\n" * 2)

    def test_reload_wrapper_until_success(self):
        main = 'import os\nimport sys\n\nif "TESTAPP_STARTED" in os.environ:\n    print("exiting cleanly")\n    sys.exit(0)\nelse:\n    print("reloading")\n    exec(open("run_twice_magic.py").read())\n'
        self.write_files({'main.py': main})
        out = self.run_subprocess([sys.executable, '-m', 'tornado.autoreload', '--until-success', 'main.py'])
        self.assertEqual(out, 'reloading\nexiting cleanly\n')