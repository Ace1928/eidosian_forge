import base64
import inspect
import builtins
class NXFlowSpecFieldType(TypeDescr):

    @staticmethod
    def encode(v):
        if not isinstance(v, tuple):
            return v
        field, ofs = v
        return [field, ofs]

    @staticmethod
    def decode(v):
        if not isinstance(v, list):
            return v
        field, ofs = v
        return (field, ofs)