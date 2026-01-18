from yaml._yaml import CParser, CEmitter
from constructor import *
from serializer import *
from representer import *
from resolver import *
class CUnsafeLoader(CParser, UnsafeConstructor, Resolver):

    def __init__(self, stream):
        CParser.__init__(self, stream)
        UnsafeConstructor.__init__(self)
        Resolver.__init__(self)