import enum
import logging
import os
import types
import typing
class SignatureDefinition(object):

    def __init__(self, name: str, rtype: str, arguments: typing.Tuple[str, ...]):
        self.name = name
        self.rtype = rtype
        self.arguments = arguments

    @property
    def callback_def(self) -> str:
        return '{} ({})'.format(self.rtype, ' ,'.join(self.arguments))

    @property
    def extern_def(self) -> str:
        return '{} {}({})'.format(self.rtype, self.name, ' ,'.join(self.arguments))

    @property
    def extern_python_def(self) -> str:
        return 'extern "Python" {} {}({});'.format(self.rtype, self.name, ' ,'.join(self.arguments))