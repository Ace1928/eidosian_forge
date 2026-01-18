from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.storage import storage_util
Gets the values of `self.files` in a deterministic order.

    Internally, `self.files` is a dictionary. Prior to Python 3.6, dictionaries
    were ordered nondeterministically, but our tests require determinism. As
    such, we have to convert the underlying dictionary to a list and sort that
    list. The specific order itself (e.g. alphabetical vs. reverse-alphabetical)
    doesn't matter.

    Returns:
      A list of files in a deterministic order.
    