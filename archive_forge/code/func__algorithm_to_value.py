from libcloud.common.base import BaseDriver, ConnectionKey
from libcloud.common.types import LibcloudError
def _algorithm_to_value(self, algorithm):
    """
        Return string value for the provided algorithm.

        :param value: Algorithm enum.
        :type  value: :class:`Algorithm`

        :rtype: ``str``
        """
    try:
        return self._ALGORITHM_TO_VALUE_MAP[algorithm]
    except KeyError:
        raise LibcloudError(value='Invalid algorithm: %s' % algorithm, driver=self)