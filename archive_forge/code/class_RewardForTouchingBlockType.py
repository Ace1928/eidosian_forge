import abc
from minerl.herobraine.hero.mc import strip_item_prefix
from minerl.herobraine.hero.spaces import Box
from minerl.herobraine.hero.handlers.translation import TranslationHandler
from minerl.herobraine.hero.handler import Handler
import jinja2
from typing import List, Dict, Union
import numpy as np
class RewardForTouchingBlockType(RewardHandler):

    def to_string(self) -> str:
        return 'reward_for_touching_block_type'

    def xml_template(self) -> str:
        return str('<RewardForTouchingBlockType>\n                    {% for block in blocks %}\n                    <Block reward="{{ block.reward }}" type="{{ block.type }}" behaviour="{{ block.behaviour }}" />\n                    {% endfor %}\n                </RewardForTouchingBlockType>')

    def __init__(self, blocks: List[Dict[str, Union[str, int, float]]]):
        """Creates a reward which is awarded when the player touches a block.
        An example of instantiating the class:

        reward = RewardForTouchingBlockType([
            {'type':'diamond_block', 'behaviour':'onceOnly', 'reward':'10'},
        ])
        """
        super().__init__()
        self.blocks = blocks
        self.fired = {bl['type']: False for bl in self.blocks}
        for block in self.blocks:
            assert set(block.keys()) == {'reward', 'type', 'behaviour'}

    def from_universal(self, obs):
        reward = 0
        if 'touched_blocks' in obs:
            for block in obs['touched_blocks']:
                for bl in self.blocks:
                    if bl['type'] in block['name'] and (not self.fired[bl['type']] or bl['behaviour'] != 'onlyOnce'):
                        reward += bl['reward']
                        self.fired[bl['type']] = True
        return reward

    def reset(self):
        self.fired = {bl['type']: False for bl in self.blocks}