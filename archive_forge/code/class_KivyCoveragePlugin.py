import os
import coverage
from kivy.lang.parser import Parser
class KivyCoveragePlugin(coverage.plugin.CoveragePlugin):

    def file_tracer(self, filename):
        if filename.endswith('.kv'):
            return KivyFileTracer(filename=filename)
        return None

    def file_reporter(self, filename):
        return KivyFileReporter(filename=filename)

    def find_executable_files(self, src_dir):
        for dirpath, dirnames, filenames in os.walk(src_dir):
            for filename in filenames:
                if filename.endswith('.kv'):
                    yield os.path.join(dirpath, filename)