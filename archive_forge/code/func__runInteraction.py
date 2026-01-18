from twisted.internet import threads
from twisted.python import log, reflect
def _runInteraction(self, interaction, *args, **kw):
    conn = self.connectionFactory(self)
    trans = self.transactionFactory(self, conn)
    try:
        result = interaction(trans, *args, **kw)
        trans.close()
        conn.commit()
        return result
    except BaseException:
        try:
            conn.rollback()
        except BaseException:
            log.err(None, 'Rollback failed')
        raise