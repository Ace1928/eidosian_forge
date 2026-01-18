from __future__ import with_statement
import os
import sys
import textwrap
import unittest
import subprocess
import tempfile
class TestTool(unittest.TestCase):
    data = '\n\n        [["blorpie"],[ "whoops" ] , [\n                                 ],\t"d-shtaeou",\r"d-nthiouh",\n        "i-vhbjkhnth", {"nifty":87}, {"morefield" :\tfalse,"field"\n            :"yes"}  ]\n           '
    expect = textwrap.dedent('    [\n        [\n            "blorpie"\n        ],\n        [\n            "whoops"\n        ],\n        [],\n        "d-shtaeou",\n        "d-nthiouh",\n        "i-vhbjkhnth",\n        {\n            "nifty": 87\n        },\n        {\n            "field": "yes",\n            "morefield": false\n        }\n    ]\n    ')

    def runTool(self, args=None, data=None):
        argv = [sys.executable, '-m', 'simplejson.tool']
        if args:
            argv.extend(args)
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        out, err = proc.communicate(data)
        self.assertEqual(strip_python_stderr(err), ''.encode())
        self.assertEqual(proc.returncode, 0)
        return out.decode('utf8').splitlines()

    def test_stdin_stdout(self):
        self.assertEqual(self.runTool(data=self.data.encode()), self.expect.splitlines())

    def test_infile_stdout(self):
        infile, infile_name = open_temp_file()
        try:
            infile.write(self.data.encode())
            infile.close()
            self.assertEqual(self.runTool(args=[infile_name]), self.expect.splitlines())
        finally:
            os.unlink(infile_name)

    def test_infile_outfile(self):
        infile, infile_name = open_temp_file()
        try:
            infile.write(self.data.encode())
            infile.close()
            outfile, outfile_name = open_temp_file()
            try:
                outfile.close()
                self.assertEqual(self.runTool(args=[infile_name, outfile_name]), [])
                with open(outfile_name, 'rb') as f:
                    self.assertEqual(f.read().decode('utf8').splitlines(), self.expect.splitlines())
            finally:
                os.unlink(outfile_name)
        finally:
            os.unlink(infile_name)