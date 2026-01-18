import unittest
from _pydev_bundle._pydev_saved_modules import thread
import queue as Queue
from _pydev_runfiles import pydev_runfiles_xml_rpc
import time
import os
import threading
import sys
class ClientThread(threading.Thread):

    def __init__(self, job_id, port, verbosity, coverage_output_file=None, coverage_include=None):
        threading.Thread.__init__(self)
        self.daemon = True
        self.port = port
        self.job_id = job_id
        self.verbosity = verbosity
        self.finished = False
        self.coverage_output_file = coverage_output_file
        self.coverage_include = coverage_include

    def _reader_thread(self, pipe, target):
        while True:
            target.write(pipe.read(1))

    def run(self):
        try:
            from _pydev_runfiles import pydev_runfiles_parallel_client
            args = [sys.executable, pydev_runfiles_parallel_client.__file__, str(self.job_id), str(self.port), str(self.verbosity)]
            if self.coverage_output_file and self.coverage_include:
                args.append(self.coverage_output_file)
                args.append(self.coverage_include)
            import subprocess
            if False:
                proc = subprocess.Popen(args, env=os.environ, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                thread.start_new_thread(self._reader_thread, (proc.stdout, sys.stdout))
                thread.start_new_thread(target=self._reader_thread, args=(proc.stderr, sys.stderr))
            else:
                proc = subprocess.Popen(args, env=os.environ, shell=False)
                proc.wait()
        finally:
            self.finished = True