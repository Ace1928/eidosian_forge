from __future__ import absolute_import
import re
from ruamel import yaml
from googlecloudsdk.third_party.appengine._internal import six_subset
class ValidatedDict(ValidatedBase, dict):
    """Base class for validated dictionaries.

  You can control the keys and values that are allowed in the dictionary
  by setting KEY_VALIDATOR and VALUE_VALIDATOR to subclasses of Validator (or
  things that can be interpreted as validators, see AsValidator).

  For example if you wanted only capitalized keys that map to integers
  you could do:

    class CapitalizedIntegerDict(ValidatedDict):
      KEY_VALIDATOR = Regex('[A-Z].*')
      VALUE_VALIDATOR = int  # this gets interpreted to Type(int)

  The following code would result in an error:

    my_dict = CapitalizedIntegerDict()
    my_dict['lowercase'] = 5  # Throws a validation exception

  You can freely nest Validated and ValidatedDict inside each other so:

    class MasterObject(Validated):
      ATTRIBUTES = {'paramdict': CapitalizedIntegerDict}

  Could be used to parse the following yaml:
    paramdict:
      ArbitraryKey: 323
      AnotherArbitraryKey: 9931
  """
    KEY_VALIDATOR = None
    VALUE_VALIDATOR = None

    def __init__(self, **kwds):
        """Construct a validated dict by interpreting the key and value validators.

    Args:
      **kwds: keyword arguments will be validated and put into the dict.
    """
        super(ValidatedDict, self).__init__()
        self.update(kwds)

    @classmethod
    def GetValidator(cls, key):
        """Check the key for validity and return a corresponding value validator.

    Args:
      key: The key that will correspond to the validator we are returning.
    """
        key = AsValidator(cls.KEY_VALIDATOR)(key, 'key in %s' % cls.__name__)
        return AsValidator(cls.VALUE_VALIDATOR)

    def __setitem__(self, key, value):
        """Set an item.

    Only attributes accepted by GetValidator and values that validate
    with the validator returned from GetValidator are allowed to be set
    in this dictionary.

    Args:
      key: Name of item to set.
      value: Items new value.

    Raises:
      ValidationError: when trying to assign to a value that does not exist.
    """
        dict.__setitem__(self, key, self.GetValidator(key)(value, key))

    def setdefault(self, key, value=None):
        """Trap setdefaultss to ensure all key/value pairs are valid.

    See the documentation for setdefault on dict for usage details.

    Raises:
      ValidationError: if the specified key is illegal or the
      value invalid.
    """
        return dict.setdefault(self, key, self.GetValidator(key)(value, key))

    def update(self, other, **kwds):
        """Trap updates to ensure all key/value pairs are valid.

    See the documentation for update on dict for usage details.

    Raises:
      ValidationError: if any of the specified keys are illegal or
        values invalid.
    """
        if hasattr(other, 'keys') and callable(getattr(other, 'keys')):
            newother = {}
            for k in other:
                newother[k] = self.GetValidator(k)(other[k], k)
        else:
            newother = [(k, self.GetValidator(k)(v, k)) for k, v in other]
        newkwds = {}
        for k in kwds:
            newkwds[k] = self.GetValidator(k)(kwds[k], k)
        dict.update(self, newother, **newkwds)

    def Set(self, key, value):
        """Set a single value on Validated instance.

    This method checks that a given key and value are valid and if so
    puts the item into this dictionary.

    Args:
      key: The name of the attributes
      value: The value to set

    Raises:
      ValidationError: when no validated attribute exists on class.
    """
        self[key] = value

    def GetWarnings(self):
        ret = []
        for name, value in self.items():
            ret.extend(self.GetValidator(name).GetWarnings(value, name, self))
        return ret

    def ToDict(self):
        """Convert ValidatedBase object to a dictionary.

    Recursively traverses all of its elements and converts everything to
    simplified collections.

    Subclasses should override this method.

    Returns:
      A dictionary mapping all attributes to simple values or collections.
    """
        result = {}
        for name, value in self.items():
            validator = self.GetValidator(name)
            result[name] = _SimplifiedValue(validator, value)
        return result