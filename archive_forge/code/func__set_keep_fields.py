from parlai.core.worlds import BatchWorld, DynamicBatchWorld
from parlai.utils.misc import msg_to_str
from parlai.utils.conversations import Conversations
import parlai.utils.logging as logging
import copy
from tqdm import tqdm
def _set_keep_fields(self, opt):
    self.keep_fields = opt['log_keep_fields'].split(',')
    self.keep_all = KEEP_ALL in self.keep_fields