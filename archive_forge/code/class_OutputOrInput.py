import ctypes,logging
from OpenGL._bytes import bytes, unicode, as_8_bit
from OpenGL._null import NULL
from OpenGL import acceleratesupport
class OutputOrInput(Output):
    DO_OUTPUT = (None, NULL)

    def __call__(self, pyArgs, index, baseOperation):
        for do_output in self.DO_OUTPUT:
            if pyArgs[index] is do_output:
                return super(OutputOrInput, self).__call__(pyArgs, index, baseOperation)
        return self.arrayType.asArray(pyArgs[index])