from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import sys
import six
def _PopulateHead(self, num_elements=1):
    """Populates self.head from the underlying iterator.

    Args:
      num_elements: Populate until self.head contains this many
          elements (or until the underlying iterator runs out).

    Returns:
      Number of elements at self.head after execution complete.
    """
    while not self.underlying_iter_empty and len(self.head) < num_elements:
        try:
            if not self.base_iterator:
                self.base_iterator = iter(self.orig_iterator)
            e = next(self.base_iterator)
            self.underlying_iter_empty = False
            if isinstance(e, tuple) and isinstance(e[0], Exception):
                self.head.append(('exception', e[0], e[1]))
            else:
                self.head.append(('element', e))
        except StopIteration:
            self.underlying_iter_empty = True
        except Exception as e:
            self.head.append(('exception', e, sys.exc_info()[2]))
    return len(self.head)