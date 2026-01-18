class TransactionFailedError(Error):
    """Raised by RunInTransaction methods when the transaction could not be
  committed, even after retrying. This is usually due to high contention.
  """