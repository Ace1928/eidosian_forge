import unittest
def _make_IOErr(self):

    class Foo:

        def __init__(self, name, *bases):
            self.__name__ = name
            self.__bases__ = bases

        def __repr__(self):
            return self.__name__
    IEx = Foo('IEx')
    IStdErr = Foo('IStdErr', IEx)
    IEnvErr = Foo('IEnvErr', IStdErr)
    IIOErr = Foo('IIOErr', IEnvErr)
    IOSErr = Foo('IOSErr', IEnvErr)
    IOErr = Foo('IOErr', IEnvErr, IIOErr, IOSErr)
    return (IOErr, [IOErr, IIOErr, IOSErr, IEnvErr, IStdErr, IEx])