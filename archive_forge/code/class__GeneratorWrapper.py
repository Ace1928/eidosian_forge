import sys
class _GeneratorWrapper:

    def __init__(self, gen):
        self.__wrapped = gen
        self.__isgen = gen.__class__ is GeneratorType
        self.__name__ = getattr(gen, '__name__', None)
        self.__qualname__ = getattr(gen, '__qualname__', None)

    def send(self, val):
        return self.__wrapped.send(val)

    def throw(self, tp, *rest):
        return self.__wrapped.throw(tp, *rest)

    def close(self):
        return self.__wrapped.close()

    @property
    def gi_code(self):
        return self.__wrapped.gi_code

    @property
    def gi_frame(self):
        return self.__wrapped.gi_frame

    @property
    def gi_running(self):
        return self.__wrapped.gi_running

    @property
    def gi_yieldfrom(self):
        return self.__wrapped.gi_yieldfrom
    cr_code = gi_code
    cr_frame = gi_frame
    cr_running = gi_running
    cr_await = gi_yieldfrom

    def __next__(self):
        return next(self.__wrapped)

    def __iter__(self):
        if self.__isgen:
            return self.__wrapped
        return self
    __await__ = __iter__