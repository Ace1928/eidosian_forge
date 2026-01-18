from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from collections import namedtuple
import errno
import json
import random
import six
import gslib
from gslib.exception import CommandException
from gslib.tracker_file import (WriteJsonDataToTrackerFile,
from gslib.utils.constants import UTF8
def _ParseLegacyTrackerData(tracker_data):
    """Parses a legacy parallel composite upload tracker file.

  Args:
    tracker_data: Legacy tracker file contents.

  Returns:
    component_prefix: The prefix used in naming the existing components, or
        None if no prefix was found.
    existing_components: A list of ObjectFromTracker objects representing
        the set of files that have already been uploaded.
  """
    old_tracker_data = tracker_data.split('\n')[:-1]
    prefix = None
    existing_components = []
    if old_tracker_data:
        prefix = old_tracker_data[0]
        i = 1
        while i < len(old_tracker_data) - 1:
            name, generation = (old_tracker_data[i], old_tracker_data[i + 1])
            if not generation:
                generation = None
            existing_components.append(ObjectFromTracker(name, generation))
            i += 2
    return (prefix, existing_components)