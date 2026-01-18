import ctypes,logging
from OpenGL._bytes import bytes, unicode, as_8_bit
from OpenGL._null import NULL
from OpenGL import acceleratesupport
class SizedOutput(Output):
    """Output generating dynamically-sized typed output arrays

        Takes an extra parameter "specifier", which is the name of
        a Python argument to be passed to the lookup function in order
        to determine the appropriate size for the output array.
        """
    argNames = ('name', 'specifier', 'lookup', 'arrayType')
    indexLookups = [('outIndex', 'name', 'cArgIndex'), ('index', 'specifier', 'pyArgIndex')]
    __slots__ = ('index', 'outIndex', 'specifier', 'lookup', 'arrayType')

    def getSize(self, pyArgs):
        """Retrieve the array size for this argument"""
        try:
            specifier = pyArgs[self.index]
        except AttributeError:
            raise RuntimeError('"Did not resolve parameter index for %r' % self.name)
        else:
            try:
                return self.lookup(specifier)
            except KeyError:
                raise KeyError('Unknown specifier %s' % specifier)