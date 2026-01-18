from ... import controldir
from ...commands import Command
from ...option import Option, RegistryOption
from . import helpers, load_fastimport
def _get_user_mapper(filename):
    from . import user_mapper
    if filename is None:
        return None
    f = open(filename)
    lines = f.readlines()
    f.close()
    return user_mapper.UserMapper(lines)