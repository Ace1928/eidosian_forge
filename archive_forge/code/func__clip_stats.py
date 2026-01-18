import gym
import json
import numpy as np
from copy import deepcopy
from minerl.herobraine.hero import mc, handlers
from collections import defaultdict, deque
def _clip_stats(self, ob):
    """
        Adjusts stats (currently, only inventory) by the amount at the end of the replay
        """
    if self.multiagent:
        return {'agent_0': subtract_stats(ob['agent_0'], self.last_ob['agent_0'])}
    else:
        return subtract_stats(ob, self.last_ob)