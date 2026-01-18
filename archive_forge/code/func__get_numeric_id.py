import datetime
from typing import Any, Dict, List, Type, Union, Iterator, Optional
from libcloud import __version__
from libcloud.dns.types import RecordType
from libcloud.common.base import BaseDriver, Connection, ConnectionUserAndKey
def _get_numeric_id(self):
    """
        Return numeric ID for the provided record if the ID is a digit.

        This method is used for sorting the values when exporting Zone to a
        BIND format.
        """
    record_id = self.id
    if record_id is None:
        return ''
    if record_id.isdigit():
        record_id_int = int(record_id)
        return record_id_int
    return record_id