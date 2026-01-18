from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import itertools
import logging
import os
import sys
from xml.dom import minidom
from absl.flags import _exceptions
from absl.flags import _flag
from absl.flags import _helpers
from absl.flags import _validators_classes
import six
def _assert_validators(self, validators):
    """Asserts if all validators in the list are satisfied.

    It asserts validators in the order they were created.

    Args:
      validators: Iterable(validators.Validator), validators to be verified.

    Raises:
      AttributeError: Raised if validators work with a non-existing flag.
      IllegalFlagValueError: Raised if validation fails for at least one
          validator.
    """
    messages = []
    bad_flags = set()
    for validator in sorted(validators, key=lambda validator: validator.insertion_index):
        try:
            if isinstance(validator, _validators_classes.SingleFlagValidator):
                if validator.flag_name in bad_flags:
                    continue
            elif isinstance(validator, _validators_classes.MultiFlagsValidator):
                if bad_flags & set(validator.flag_names):
                    continue
            validator.verify(self)
        except _exceptions.ValidationError as e:
            if isinstance(validator, _validators_classes.SingleFlagValidator):
                bad_flags.add(validator.flag_name)
            elif isinstance(validator, _validators_classes.MultiFlagsValidator):
                bad_flags.update(set(validator.flag_names))
            message = validator.print_flags_with_values(self)
            messages.append('%s: %s' % (message, str(e)))
    if messages:
        raise _exceptions.IllegalFlagValueError('\n'.join(messages))