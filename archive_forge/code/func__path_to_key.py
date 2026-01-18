from . import errors, osutils
@staticmethod
def _path_to_key(path):
    dirname, basename = osutils.split(path)
    return (dirname.split('/'), basename)