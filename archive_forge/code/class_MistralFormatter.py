import abc
import datetime as dt
import textwrap
from osc_lib.command import command
class MistralFormatter(metaclass=abc.ABCMeta):
    COLUMNS = []

    @classmethod
    def fields(cls):
        return [c[0] for c in cls.COLUMNS if len(c) == 2 or not c[2]]

    @classmethod
    def headings(cls):
        return [c[1] for c in cls.COLUMNS]

    @classmethod
    def format_list(cls, instance=None):
        return cls.format(instance, lister=True)

    @staticmethod
    def format(instance=None, lister=False):
        pass