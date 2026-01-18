from tensorflow.python.util.tf_export import tf_export
from tensorflow.python import pywrap_tfe
def _as_int_or_none(inp):
    return None if inp is None else int(inp)