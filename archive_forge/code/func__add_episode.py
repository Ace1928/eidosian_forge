from parlai.core.worlds import BatchWorld, DynamicBatchWorld
from parlai.utils.misc import msg_to_str
from parlai.utils.conversations import Conversations
import parlai.utils.logging as logging
import copy
from tqdm import tqdm
def _add_episode(self, episode):
    """
        Add episode to the logs.
        """
    self._logs.append(episode)