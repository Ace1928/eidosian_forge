import logging as _logging
import sys as _sys
from absl.flags import *  # pylint: disable=wildcard-import
from tensorflow.python.util import tf_decorator
class _FlagValuesWrapper:
    """Wrapper class for absl.flags.FLAGS.

  The difference is that tf.flags.FLAGS implicitly parses flags with sys.argv
  when accessing the FLAGS values before it's explicitly parsed,
  while absl.flags.FLAGS raises an exception.
  """

    def __init__(self, flags_object):
        self.__dict__['__wrapped'] = flags_object

    def __getattribute__(self, name):
        if name == '__dict__':
            return super().__getattribute__(name)
        return self.__dict__['__wrapped'].__getattribute__(name)

    def __getattr__(self, name):
        wrapped = self.__dict__['__wrapped']
        if not wrapped.is_parsed():
            wrapped(_sys.argv)
        return wrapped.__getattr__(name)

    def __setattr__(self, name, value):
        return self.__dict__['__wrapped'].__setattr__(name, value)

    def __delattr__(self, name):
        return self.__dict__['__wrapped'].__delattr__(name)

    def __dir__(self):
        return self.__dict__['__wrapped'].__dir__()

    def __getitem__(self, name):
        return self.__dict__['__wrapped'].__getitem__(name)

    def __setitem__(self, name, flag):
        return self.__dict__['__wrapped'].__setitem__(name, flag)

    def __len__(self):
        return self.__dict__['__wrapped'].__len__()

    def __iter__(self):
        return self.__dict__['__wrapped'].__iter__()

    def __str__(self):
        return self.__dict__['__wrapped'].__str__()

    def __call__(self, *args, **kwargs):
        return self.__dict__['__wrapped'].__call__(*args, **kwargs)