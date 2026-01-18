from typing import Optional, Set
class ExceptionWrapper:

    def __init__(self, exception_class):
        self.exception_class = exception_class

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and (not isinstance(exc_val, self.exception_class)):
            raise self.exception_class(str(exc_val)) from exc_val
        return False