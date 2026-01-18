from parlai.core.worlds import BatchWorld, DynamicBatchWorld
from parlai.utils.misc import msg_to_str
from parlai.utils.conversations import Conversations
import parlai.utils.logging as logging
import copy
from tqdm import tqdm
def _add_msgs(self, acts, idx=0):
    """
        Add messages from a `parley()` to the current episode of logs.

        :param acts: list of acts from a `.parley()` call
        """
    msgs = []
    for act in acts:
        if not self.keep_all:
            msg = {f: act[f] for f in self.keep_fields if f in act}
        else:
            msg = act
        msgs.append(msg)
    self._current_episodes.setdefault(idx, [])
    self._current_episodes[idx].append(msgs)