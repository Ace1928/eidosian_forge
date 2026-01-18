from .rfc822 import _parse_date_rfc822
def _parse_date_asctime(dt):
    """Parse asctime-style dates.

    Converts asctime to RFC822-compatible dates and uses the RFC822 parser
    to do the actual parsing.

    Supported formats (format is standardized to the first one listed):

    * {weekday name} {month name} dd hh:mm:ss {+-tz} yyyy
    * {weekday name} {month name} dd hh:mm:ss yyyy
    """
    parts = dt.split()
    if len(parts) == 5:
        parts.insert(4, '+0000')
    if len(parts) != 6:
        return None
    return _parse_date_rfc822(' '.join([parts[0], parts[2], parts[1], parts[5], parts[3], parts[4]]))