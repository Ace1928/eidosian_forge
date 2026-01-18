import datetime
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
class DateTimeField(messages.MessageField):
    """Field definition for datetime values.

    Stores a python datetime object as a field.  If time zone information is
    included in the datetime object, it will be included in
    the encoded data when this is encoded/decoded.
    """
    type = datetime.datetime
    message_type = DateTimeMessage

    @util.positional(3)
    def __init__(self, number, **kwargs):
        super(DateTimeField, self).__init__(self.message_type, number, **kwargs)

    def value_from_message(self, message):
        """Convert DateTimeMessage to a datetime.

        Args:
          A DateTimeMessage instance.

        Returns:
          A datetime instance.
        """
        message = super(DateTimeField, self).value_from_message(message)
        if message.time_zone_offset is None:
            return datetime.datetime.utcfromtimestamp(message.milliseconds / 1000.0)
        milliseconds = message.milliseconds - 60000 * message.time_zone_offset
        timezone = util.TimeZoneOffset(message.time_zone_offset)
        return datetime.datetime.fromtimestamp(milliseconds / 1000.0, tz=timezone)

    def value_to_message(self, value):
        value = super(DateTimeField, self).value_to_message(value)
        if value.tzinfo is None:
            time_zone_offset = 0
            local_epoch = datetime.datetime.utcfromtimestamp(0)
        else:
            time_zone_offset = util.total_seconds(value.tzinfo.utcoffset(value))
            local_epoch = datetime.datetime.fromtimestamp(-time_zone_offset, tz=value.tzinfo)
        delta = value - local_epoch
        message = DateTimeMessage()
        message.milliseconds = int(util.total_seconds(delta) * 1000)
        if value.tzinfo is not None:
            utc_offset = value.tzinfo.utcoffset(value)
            if utc_offset is not None:
                message.time_zone_offset = int(util.total_seconds(value.tzinfo.utcoffset(value)) / 60)
        return message