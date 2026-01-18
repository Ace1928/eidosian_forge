import sys, os
import textwrap
def _check_choice(self):
    if self.type == 'choice':
        if self.choices is None:
            raise OptionError("must supply a list of choices for type 'choice'", self)
        elif not isinstance(self.choices, (tuple, list)):
            raise OptionError("choices must be a list of strings ('%s' supplied)" % str(type(self.choices)).split("'")[1], self)
    elif self.choices is not None:
        raise OptionError('must not supply choices for type %r' % self.type, self)