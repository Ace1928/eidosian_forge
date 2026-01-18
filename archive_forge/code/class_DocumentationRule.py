from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentationRule(_messages.Message):
    """A documentation rule provides information about individual API elements.

  Fields:
    deprecationDescription: Deprecation description of the selected
      element(s). It can be provided if an element is marked as `deprecated`.
    description: Description of the selected API(s).
    selector: The selector is a comma-separated list of patterns. Each pattern
      is a qualified name of the element which may end in "*", indicating a
      wildcard. Wildcards are only allowed at the end and for a whole
      component of the qualified name, i.e. "foo.*" is ok, but not "foo.b*" or
      "foo.*.bar". To specify a default for all applicable elements, the whole
      pattern "*" is used.
  """
    deprecationDescription = _messages.StringField(1)
    description = _messages.StringField(2)
    selector = _messages.StringField(3)