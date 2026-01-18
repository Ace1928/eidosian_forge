import time
from io import BytesIO
from ... import errors as bzr_errors
from ... import tests
from ...tests.features import Feature, ModuleAvailableFeature
from .. import import_dulwich
@staticmethod
def _encode_path(path):
    if isinstance(path, bytes):
        return path
    if '\n' in path or path[0] == '"':
        path = path.replace('\\', '\\\\')
        path = path.replace('\n', '\\n')
        path = path.replace('"', '\\"')
        path = '"' + path + '"'
    return path.encode('utf-8')