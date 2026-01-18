import base64
import inspect
import builtins
@classmethod
def _encode_value(cls, k, v, encode_string=base64.b64encode):
    return cls._get_encoder(k, encode_string)(v)