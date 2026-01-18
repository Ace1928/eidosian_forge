import os
import warnings
from enchant.errors import Error, DictNotFoundError
from enchant.utils import get_default_language
from enchant.pypwl import PyPWL
def __describe_dict(self, dict_data):
    """Get the description tuple for a dict data object.
        <dict_data> must be a C-library pointer to an enchant dictionary.
        The return value is a tuple of the form:
                (<tag>,<name>,<desc>,<file>)
        """
    cb_result = []

    def cb_func(tag, name, desc, file):
        tag = tag.decode()
        name = name.decode()
        desc = desc.decode()
        file = file.decode()
        cb_result.append((tag, name, desc, file))
    _e.dict_describe(dict_data, cb_func)
    return cb_result[0]