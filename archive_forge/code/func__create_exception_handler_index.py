from ..construct import UBInt32, ULInt32, Struct
def _create_exception_handler_index(self):
    self.EH_index_struct = Struct('EH_index', self.EHABI_uint32('word0'), self.EHABI_uint32('word1'))