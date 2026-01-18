from minerl.herobraine.hero.handler import Handler
from typing import Dict, List, Union
import random
import jinja2
class SimpleInventoryAgentStart(InventoryAgentStart):
    """ An inventory agentstart specification which
    just fills the inventory of the agent sequentially.
    """

    def __init__(self, inventory: List[Dict[str, Union[str, int]]]):
        """ Creates a simple inventory agent start.

        For example:

            sias =  SimpleInventoryAgentStart(
                [
                    {'type':'dirt', 'quantity':10},
                    {'type':'planks', 'quantity':5},
                    {'type':'log', 'quantity':1},
                    {'type':'iron_ore', 'quantity':4}
                ]
            )
        """
        super().__init__({i: item for i, item in enumerate(inventory)})