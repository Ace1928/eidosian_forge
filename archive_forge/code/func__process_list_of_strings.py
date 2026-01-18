import ast
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
def _process_list_of_strings(self, names):
    for i in range(len(names)):
        qn = qual_names.QN(names[i])
        if qn in self.name_map:
            names[i] = str(self.name_map[qn])
    return names