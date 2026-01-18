from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from six.moves import zip_longest
@classmethod
def _FromString(cls, version):
    """Parse the given version string into its parts."""
    if version is None:
        raise ParseError('The value is not a valid SemVer string: [None]')
    try:
        match = re.match(_SEMVER, version)
    except (TypeError, re.error) as e:
        raise ParseError('Error parsing version string: [{0}].  {1}'.format(version, e))
    if not match:
        raise ParseError('The value is not a valid SemVer string: [{0}]'.format(version))
    parts = match.groupdict()
    return (int(parts['major']), int(parts['minor']), int(parts['patch']), parts['prerelease'], parts['build'])