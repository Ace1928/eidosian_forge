from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import with_statement
import re
import zipfile
def _ParseManifestSection(section, jar_file_name):
    """Parse a dict out of the given manifest section string.

  Args:
    section: a str or unicode that is the manifest section. It looks something
      like this (without the >):
      > Name: section-name
      > Some-Attribute: some value
      > Another-Attribute: another value
    jar_file_name: a str that is the path of the jar, for use in exception
      messages.

  Returns:
    A dict where the keys are the attributes (here, 'Name', 'Some-Attribute',
    'Another-Attribute'), and the values are the corresponding attribute values.

  Raises:
    InvalidJarError: if the manifest section is not well-formed.
  """
    section = section.replace('\n ', '').rstrip('\n')
    if not section:
        return {}
    try:
        return dict((line.split(': ', 1) for line in section.split('\n')))
    except ValueError:
        raise InvalidJarError('%s: Invalid manifest %r' % (jar_file_name, section))