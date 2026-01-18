from typing import Optional
class ReadOnlyMailbox(MailboxException):

    def __str__(self) -> str:
        return 'Mailbox open in read-only state'