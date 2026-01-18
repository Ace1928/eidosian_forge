import warnings
from .api import _, is_validator, FancyValidator, Invalid, NoDefault
from . import declarative
from .exc import FERuntimeWarning
class SimpleFormValidator(FancyValidator):
    """
    This validator wraps a simple function that validates the form.

    The function looks something like this::

      >>> def validate(form_values, state, validator):
      ...     if form_values.get('country', 'US') == 'US':
      ...         if not form_values.get('state'):
      ...             return dict(state='You must enter a state')
      ...     if not form_values.get('country'):
      ...         form_values['country'] = 'US'

    This tests that the field 'state' must be filled in if the country
    is US, and defaults that country value to 'US'.  The ``validator``
    argument is the SimpleFormValidator instance, which you can use to
    format messages or keep configuration state in if you like (for
    simple ad hoc validation you are unlikely to need it).

    To create a validator from that function, you would do::

      >>> from formencode.schema import SimpleFormValidator
      >>> validator = SimpleFormValidator(validate)
      >>> validator.to_python({'country': 'US', 'state': ''}, None)
      Traceback (most recent call last):
          ...
      Invalid: state: You must enter a state
      >>> sorted(validator.to_python({'state': 'IL'}, None).items())
      [('country', 'US'), ('state', 'IL')]

    The validate function can either return a single error message
    (that applies to the whole form), a dictionary that applies to the
    fields, None which means the form is valid, or it can raise
    Invalid.

    Note that you may update the value_dict *in place*, but you cannot
    return a new value.

    Another way to instantiate a validator is like this::

      >>> @SimpleFormValidator.decorate()
      ... def MyValidator(value_dict, state):
      ...     return None # or some more useful validation

    After this ``MyValidator`` will be a ``SimpleFormValidator``
    instance (it won't be your function).
    """
    __unpackargs__ = ('func',)
    validate_partial_form = False

    def __initargs__(self, new_attrs):
        self.__doc__ = getattr(self.func, '__doc__', None)

    def to_python(self, value_dict, state):
        value_dict = value_dict.copy()
        errors = self.func(value_dict, state, self)
        if not errors:
            return value_dict
        if isinstance(errors, str):
            raise Invalid(errors, value_dict, state)
        if isinstance(errors, dict):
            raise Invalid(format_compound_error(errors), value_dict, state, error_dict=errors)
        if isinstance(errors, Invalid):
            raise errors
        raise TypeError('Invalid error value: %r' % errors)
    validate_partial = to_python

    @classmethod
    def decorate(cls, **kw):

        def decorator(func):
            return cls(func, **kw)
        return decorator