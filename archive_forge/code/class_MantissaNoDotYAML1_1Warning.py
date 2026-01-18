from __future__ import absolute_import
import warnings
import textwrap
from ruamel.yaml.compat import utf8
class MantissaNoDotYAML1_1Warning(YAMLWarning):

    def __init__(self, node, flt_str):
        self.node = node
        self.flt = flt_str

    def __str__(self):
        line = self.node.start_mark.line
        col = self.node.start_mark.column
        return '\nIn YAML 1.1 floating point values should have a dot (\'.\') in their mantissa.\nSee the Floating-Point Language-Independent Type for YAML™ Version 1.1 specification\n( http://yaml.org/type/float.html ). This dot is not required for JSON nor for YAML 1.2\n\nCorrect your float: "{}" on line: {}, column: {}\n\nor alternatively include the following in your code:\n\n  import warnings\n  warnings.simplefilter(\'ignore\', ruamel.yaml.error.MantissaNoDotYAML1_1Warning)\n\n'.format(self.flt, line, col)