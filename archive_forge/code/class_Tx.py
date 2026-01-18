from collections import namedtuple
class Tx:
    """AMQ Tx class."""
    CLASS_ID = 90
    Select = (90, 10)
    SelectOk = (90, 11)
    Commit = (90, 20)
    CommitOk = (90, 21)
    Rollback = (90, 30)
    RollbackOk = (90, 31)