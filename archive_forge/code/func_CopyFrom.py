def CopyFrom(self, other_msg):
    """Copies the content of the specified message into the current message.

    The method clears the current message and then merges the specified
    message using MergeFrom.

    Args:
      other_msg (Message): A message to copy into the current one.
    """
    if self is other_msg:
        return
    self.Clear()
    self.MergeFrom(other_msg)