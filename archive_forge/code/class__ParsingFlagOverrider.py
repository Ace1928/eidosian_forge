import collections
import functools
import inspect
from typing import overload, Any, Callable, Mapping, Tuple, TypeVar, Type, Sequence, Union
from absl import flags
class _ParsingFlagOverrider(_FlagOverrider):
    """Context manager for overriding flags.

  Simulates command line parsing.

  This is simlar to _FlagOverrider except that all **overrides should be
  strings or sequences of strings, and when context is entered this class calls
  .parse(value)

  This results in the flags having .present set properly.
  """

    def __init__(self, **overrides: Union[str, Sequence[str]]):
        for flag_name, new_value in overrides.items():
            if isinstance(new_value, str):
                continue
            if isinstance(new_value, collections.abc.Sequence) and all((isinstance(single_value, str) for single_value in new_value)):
                continue
            raise TypeError(f'flagsaver.as_parsed() cannot parse {flag_name}. Expected a single string or sequence of strings but {type(new_value)} was provided.')
        super().__init__(**overrides)

    def __enter__(self):
        self._saved_flag_values = save_flag_values(FLAGS)
        try:
            for flag_name, unparsed_value in self._overrides.items():
                FLAGS[flag_name].parse(unparsed_value)
                FLAGS[flag_name].using_default_value = False
            for flag_name in self._overrides:
                FLAGS._assert_validators(FLAGS[flag_name].validators)
        except KeyError as e:
            restore_flag_values(self._saved_flag_values, FLAGS)
            raise flags.UnrecognizedFlagError('Unknown command line flag.') from e
        except:
            restore_flag_values(self._saved_flag_values, FLAGS)
            raise