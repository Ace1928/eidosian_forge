import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
class FieldsMatch(FormValidator):
    """
    Tests that the given fields match, i.e., are identical.  Useful
    for password+confirmation fields.  Pass the list of field names in
    as `field_names`.

    ::

        >>> f = FieldsMatch('pass', 'conf')
        >>> sorted(f.to_python({'pass': 'xx', 'conf': 'xx'}).items())
        [('conf', 'xx'), ('pass', 'xx')]
        >>> f.to_python({'pass': 'xx', 'conf': 'yy'})
        Traceback (most recent call last):
            ...
        Invalid: conf: Fields do not match
    """
    show_match = False
    field_names = None
    validate_partial_form = True
    __unpackargs__ = ('*', 'field_names')
    messages = dict(invalid=_('Fields do not match (should be %(match)s)'), invalidNoMatch=_('Fields do not match'), notDict=_('Fields should be a dictionary'))

    def __init__(self, *args, **kw):
        super(FieldsMatch, self).__init__(*args, **kw)
        if len(self.field_names) < 2:
            raise TypeError('FieldsMatch() requires at least two field names')

    def validate_partial(self, field_dict, state):
        for name in self.field_names:
            if name not in field_dict:
                return
        self._validate_python(field_dict, state)

    def _validate_python(self, field_dict, state):
        try:
            ref = field_dict[self.field_names[0]]
        except TypeError:
            raise Invalid(self.message('notDict', state), field_dict, state)
        except KeyError:
            ref = ''
        errors = {}
        for name in self.field_names[1:]:
            if field_dict.get(name, '') != ref:
                if self.show_match:
                    errors[name] = self.message('invalid', state, match=ref)
                else:
                    errors[name] = self.message('invalidNoMatch', state)
        if errors:
            error_message = '<br>\n'.join(('%s: %s' % (name, value) for name, value in sorted(errors.items())))
            raise Invalid(error_message, field_dict, state, error_dict=errors)