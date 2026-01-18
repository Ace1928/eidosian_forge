import functools
import kimpy
from .exceptions import KIMModelNotFound, KIMModelInitializationError, KimpyError
class ModelCollections:
    """
    KIM Portable Models and Simulator Models are installed/managed into
    different "collections".  In order to search through the different
    KIM API model collections on the system, a corresponding object must
    be instantiated.  For more on model collections, see the KIM API's
    install file:
    https://github.com/openkim/kim-api/blob/master/INSTALL
    """

    def __init__(self):
        self.collection = collections_create()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        pass

    def get_item_type(self, model_name):
        try:
            model_type = check_call(self.collection.get_item_type, model_name)
        except KimpyError:
            msg = 'Could not find model {} installed in any of the KIM API model collections on this system.  See https://openkim.org/doc/usage/obtaining-models/ for instructions on installing models.'.format(model_name)
            raise KIMModelNotFound(msg)
        return model_type

    @property
    def initialized(self):
        return hasattr(self, 'collection')