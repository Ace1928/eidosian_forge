from textwrap import dedent
from pythran.tables import pythran_ward
from pythran.spec import signatures_to_string
from pythran.utils import quote_cxxstring
def add_global_var(self, name, init):
    self.global_vars.append(name)
    self.python_implems.append(Assign('static PyObject* ' + name, 'to_python({})'.format(init)))