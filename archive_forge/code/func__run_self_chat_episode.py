from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent, create_agent_from_model_file
from parlai.core.worlds import create_task
from parlai.utils.world_logging import WorldLogger
from parlai.utils.misc import TimeLogger
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging
import math
import json
import random
def _run_self_chat_episode(opt, world, world_logger):
    bsz = opt.get('batchsize', 1)
    num_turns = opt['selfchat_max_turns']
    num_parleys = math.ceil(num_turns / bsz)
    for _ in range(num_parleys):
        world.parley()
        world_logger.log(world)
        if opt['display_examples']:
            print(world.display())
    if opt['display_examples']:
        print('-- end of episode --')
    world.reset()
    world_logger.reset_world()