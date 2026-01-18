from jupyter_lsp.specs.r_languageserver import RLanguageServer
from jupyter_lsp.specs.utils import PythonModuleSpec
class NonInstalledPythonServer(PythonModuleSpec):
    python_module = 'not_installed_python_module'
    key = 'a_module'