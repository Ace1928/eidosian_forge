from calendar import timegm
from decimal import Decimal as MyDecimal, ROUND_HALF_EVEN
from email.utils import formatdate
import six
from flask_restful import marshal
from flask import url_for, request
def _rfc822(dt):
    """Turn a datetime object into a formatted date.

    Example::

        fields._rfc822(datetime(2011, 1, 1)) => "Sat, 01 Jan 2011 00:00:00 -0000"

    :param dt: The datetime to transform
    :type dt: datetime
    :return: A RFC 822 formatted date string
    """
    return formatdate(timegm(dt.utctimetuple()))