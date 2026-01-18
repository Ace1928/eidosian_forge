import types
import weakref
import six
from apitools.base.protorpclite import util
def __validate(self, value, validate_element):
    """Internal validation function.

        Validate an internal value using a function to validate
        individual elements.

        Args:
          value: Value to validate.
          validate_element: Function to use to validate individual elements.

        Raises:
          ValidationError if value is not expected type.

        """
    if not self.repeated:
        return validate_element(value)
    elif isinstance(value, (list, tuple)):
        result = []
        for element in value:
            if element is None:
                try:
                    name = self.name
                except AttributeError:
                    raise ValidationError('Repeated values for %s may not be None' % self.__class__.__name__)
                else:
                    raise ValidationError('Repeated values for field %s may not be None' % name)
            result.append(validate_element(element))
        return result
    elif value is not None:
        try:
            name = self.name
        except AttributeError:
            raise ValidationError('%s is repeated. Found: %s' % (self.__class__.__name__, value))
        else:
            raise ValidationError('Field %s is repeated. Found: %s' % (name, value))
    return value