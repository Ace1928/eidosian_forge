from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import datetime
import io
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import times
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import urllib
def TransformColor(r, red=None, yellow=None, green=None, blue=None, **kwargs):
    """Colorizes the resource string value.

  The *red*, *yellow*, *green* and *blue* args are RE patterns, matched against
  the resource in order. The first pattern that matches colorizes the matched
  substring with that color, and the other patterns are skipped.

  Args:
    r: A JSON-serializable object.
    red: The substring pattern for the color red.
    yellow: The substring pattern for the color yellow.
    green: The substring pattern for the color green.
    blue: The substring pattern for the color blue.
    **kwargs: console_attr.Colorizer() kwargs.

  Returns:
    A console_attr.Colorizer() object if any color substring matches, r
    otherwise.

  Example:
    `color(red=STOP,yellow=CAUTION,green=GO)`:::
    For the resource string "CAUTION means GO FASTER" displays the
    substring "CAUTION" in yellow.
  """
    string = six.text_type(r)
    for color, pattern in (('red', red), ('yellow', yellow), ('green', green), ('blue', blue)):
        if pattern and re.search(pattern, string):
            return console_attr.Colorizer(string, color, **kwargs)
    return string