from abc import ABC, abstractmethod
from .abstract import Type
from .. import types, errors
class CompileResultWAP(WrapperAddressProtocol):
    """Wrapper of dispatcher instance compilation result to turn it a
    first-class function.
    """

    def __init__(self, cres):
        """
        Parameters
        ----------
        cres : CompileResult
          Specify compilation result of a Numba jit-decorated function
          (that is a value of dispatcher instance ``overloads``
          attribute)
        """
        self.cres = cres
        name = getattr(cres.fndesc, 'llvm_cfunc_wrapper_name')
        self.address = cres.library.get_pointer_to_function(name)

    def dump(self, tab=''):
        print(f'{tab}DUMP {type(self).__name__} [addr={self.address}]')
        self.cres.signature.dump(tab=tab + '  ')
        print(f'{tab}END DUMP {type(self).__name__}')

    def __wrapper_address__(self):
        return self.address

    def signature(self):
        return self.cres.signature

    def __call__(self, *args, **kwargs):
        return self.cres.entry_point(*args, **kwargs)