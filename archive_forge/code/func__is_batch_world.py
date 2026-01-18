from parlai.core.worlds import BatchWorld, DynamicBatchWorld
from parlai.utils.misc import msg_to_str
from parlai.utils.conversations import Conversations
import parlai.utils.logging as logging
import copy
from tqdm import tqdm
def _is_batch_world(self, world):
    return (isinstance(world, BatchWorld) or isinstance(world, DynamicBatchWorld)) and len(world.worlds) > 1