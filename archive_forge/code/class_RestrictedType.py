import dill
class RestrictedType:

    def __bool__(*args, **kwargs):
        raise Exception('Restricted function')
    __eq__ = __lt__ = __le__ = __ne__ = __gt__ = __ge__ = __hash__ = __bool__