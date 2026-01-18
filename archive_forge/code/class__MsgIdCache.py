import collections
import uuid
from oslo_config import cfg
from oslo_messaging._drivers import common as rpc_common
class _MsgIdCache(object):
    """This class checks any duplicate messages."""
    DUP_MSG_CHECK_SIZE = 16

    def __init__(self, **kwargs):
        self.prev_msgids = collections.deque([], maxlen=self.DUP_MSG_CHECK_SIZE)

    def check_duplicate_message(self, message_data):
        """AMQP consumers may read same message twice when exceptions occur
           before ack is returned. This method prevents doing it.
        """
        try:
            msg_id = message_data.pop(UNIQUE_ID)
        except KeyError:
            return
        if msg_id in self.prev_msgids:
            raise rpc_common.DuplicateMessageError(msg_id=msg_id)
        return msg_id

    def add(self, msg_id):
        if msg_id and msg_id not in self.prev_msgids:
            self.prev_msgids.append(msg_id)