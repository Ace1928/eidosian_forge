from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SMTPDelivery(_messages.Message):
    """SMTPDelivery is the delivery configuration for an SMTP (email)
  notification.

  Fields:
    fromAddress: This is the SMTP account/email that appears in the `From:` of
      the email. If empty, it is assumed to be sender.
    password: The SMTP sender's password.
    port: The SMTP port of the server.
    recipientAddresses: This is the list of addresses to which we send the
      email (i.e. in the `To:` of the email).
    senderAddress: This is the SMTP account/email that is used to send the
      message.
    server: The address of the SMTP server.
  """
    fromAddress = _messages.StringField(1)
    password = _messages.MessageField('NotifierSecretRef', 2)
    port = _messages.StringField(3)
    recipientAddresses = _messages.StringField(4, repeated=True)
    senderAddress = _messages.StringField(5)
    server = _messages.StringField(6)