class OverQuotaError(Error):
    """Raised by APIProxy calls when they have been blocked due to a lack of
  available quota."""