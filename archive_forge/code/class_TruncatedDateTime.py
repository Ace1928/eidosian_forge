import netaddr
from sqlalchemy import types
from neutron_lib._i18n import _
class TruncatedDateTime(types.TypeDecorator):
    """Truncates microseconds.

    Use this for datetime fields so we don't have to worry about DB-specific
    behavior when it comes to rounding/truncating microseconds off of
    timestamps.
    """
    impl = types.DateTime
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return value.replace(microsecond=0) if value else value
    process_result_value = process_bind_param