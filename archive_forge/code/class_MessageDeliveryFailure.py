class MessageDeliveryFailure(MessagingException):
    """Raised if message sending failed after the asked retry."""