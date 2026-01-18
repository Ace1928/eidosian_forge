from .rfc822 import _parse_date_rfc822
Parse asctime-style dates.

    Converts asctime to RFC822-compatible dates and uses the RFC822 parser
    to do the actual parsing.

    Supported formats (format is standardized to the first one listed):

    * {weekday name} {month name} dd hh:mm:ss {+-tz} yyyy
    * {weekday name} {month name} dd hh:mm:ss yyyy
    