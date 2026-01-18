from boto.exception import BotoServerError
class ResponseErrorFactory(BotoServerError):

    def __new__(cls, *args, **kw):
        error = BotoServerError(*args, **kw)
        newclass = globals().get(error.error_code, ResponseError)
        obj = newclass.__new__(newclass, *args, **kw)
        obj.__dict__.update(error.__dict__)
        return obj