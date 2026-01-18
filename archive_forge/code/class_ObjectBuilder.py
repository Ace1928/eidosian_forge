from __future__ import absolute_import
from ruamel import yaml
from googlecloudsdk.third_party.appengine.api import validation
from googlecloudsdk.third_party.appengine.api import yaml_builder
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_listener
class ObjectBuilder(yaml_builder.Builder):
    """Builder used for constructing validated objects.

  Given a class that implements validation.ValidatedBase, it will parse a YAML
  document and attempt to build an instance of the class.
  ObjectBuilder will only map YAML fields that are accepted by the
  ValidatedBase's GetValidator function.
  Lists are mapped to validated.  Repeated attributes and maps are mapped to
  validated.Type properties.

  For a YAML map to be compatible with a class, the class must have a
  constructor that can be called with no parameters.  If the provided type
  does not have such a constructor a parse time error will occur.
  """

    def __init__(self, default_class):
        """Initialize validated object builder.

    Args:
      default_class: Class that is instantiated upon the detection of a new
        document.  An instance of this class will act as the document itself.
    """
        self.default_class = default_class

    def _GetRepeated(self, attribute):
        """Get the ultimate type of a repeated validator.

    Looks for an instance of validation.Repeated, returning its constructor.

    Args:
      attribute: Repeated validator attribute to find type for.

    Returns:
      The expected class of of the Type validator, otherwise object.
    """
        if isinstance(attribute, validation.Optional):
            attribute = attribute.validator
        if isinstance(attribute, validation.Repeated):
            return attribute.constructor
        return object

    def BuildDocument(self):
        """Instantiate new root validated object.

    Returns:
      New instance of validated object.
    """
        return self.default_class()

    def BuildMapping(self, top_value):
        """New instance of object mapper for opening map scope.

    Args:
      top_value: Parent of nested object.

    Returns:
      New instance of object mapper.
    """
        result = _ObjectMapper()
        if isinstance(top_value, self.default_class):
            result.value = top_value
        return result

    def EndMapping(self, top_value, mapping):
        """When leaving scope, makes sure new object is initialized.

    This method is mainly for picking up on any missing required attributes.

    Args:
      top_value: Parent of closing mapping object.
      mapping: _ObjectMapper instance that is leaving scope.
    """
        if not hasattr(mapping.value, 'CheckInitialized'):
            raise validation.ValidationError('Cannot convert map to non-map value.')
        try:
            mapping.value.CheckInitialized()
        except validation.ValidationError:
            raise
        except Exception as e:
            try:
                error_str = str(e)
            except Exception:
                error_str = '<unknown>'
            raise validation.ValidationError(error_str, e)

    def BuildSequence(self, top_value):
        """New instance of object sequence.

    Args:
      top_value: Object that contains the new sequence.

    Returns:
      A new _ObjectSequencer instance.
    """
        return _ObjectSequencer()

    def MapTo(self, subject, key, value):
        """Map key-value pair to an objects attribute.

    Args:
      subject: _ObjectMapper of object that will receive new attribute.
      key: Key of attribute.
      value: Value of new attribute.

    Raises:
      UnexpectedAttribute when the key is not a validated attribute of
      the subject value class.
    """
        assert isinstance(subject.value, validation.ValidatedBase)
        try:
            attribute = subject.value.GetValidator(key)
        except validation.ValidationError as err:
            raise yaml_errors.UnexpectedAttribute(err)
        if isinstance(value, _ObjectMapper):
            value.set_value(attribute.expected_type())
            value = value.value
        elif isinstance(value, _ObjectSequencer):
            value.set_constructor(self._GetRepeated(attribute))
            value = value.value
        subject.see(key)
        try:
            subject.value.Set(key, value)
        except validation.ValidationError as e:
            try:
                error_str = str(e)
            except Exception:
                error_str = '<unknown>'
            try:
                value_str = str(value)
            except Exception:
                value_str = '<unknown>'
            e.message = "Unable to assign value '%s' to attribute '%s':\n%s" % (value_str, key, error_str)
            raise e
        except Exception as e:
            try:
                error_str = str(e)
            except Exception:
                error_str = '<unknown>'
            try:
                value_str = str(value)
            except Exception:
                value_str = '<unknown>'
            message = "Unable to assign value '%s' to attribute '%s':\n%s" % (value_str, key, error_str)
            raise validation.ValidationError(message, e)

    def AppendTo(self, subject, value):
        """Append a value to a sequence.

    Args:
      subject: _ObjectSequence that is receiving new value.
      value: Value that is being appended to sequence.
    """
        if isinstance(value, _ObjectMapper):
            value.set_value(subject.constructor())
            subject.value.append(value.value)
        else:
            subject.value.append(value)