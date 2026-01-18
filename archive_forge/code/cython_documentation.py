from types import SimpleNamespace
 Exports a no-op 'cython' namespace similar to
https://github.com/cython/cython/blob/master/Cython/Shadow.py

This allows to optionally compile @cython decorated functions
(when cython is available at built time), or run the same code
as pure-python, without runtime dependency on cython module.

We only define the symbols that we use. E.g. see fontTools.cu2qu
