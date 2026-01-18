from yaml._yaml import CParser, CEmitter
from constructor import *
from serializer import *
from representer import *
from resolver import *
class CFullLoader(CParser, FullConstructor, Resolver):

    def __init__(self, stream):
        CParser.__init__(self, stream)
        FullConstructor.__init__(self)
        Resolver.__init__(self)