from minerl.herobraine.hero.handler import Handler
from typing import Dict, List, Union
import random
import jinja2
class InventoryAgentStart(Handler):

    def to_string(self) -> str:
        return 'inventory_agent_start'

    def xml_template(self) -> str:
        return str('<Inventory>\n            {% for  slot in inventory %}\n                <InventoryObject slot="{{ slot }}" type="{{ inventory[slot][\'type\'] }}" quantity="{{ inventory[slot][\'quantity\'] }}"/>\n            {% endfor %}\n            </Inventory>\n            ')

    def __init__(self, inventory: Dict[int, Dict[str, Union[str, int]]]):
        """Creates an inventory agent start which sets the inventory of the
        agent by slot id.

        For example:

            ias = InventoryAgentStart(
            {
                0: {'type':'dirt', 'quantity':10},
                1: {'type':'planks', 'quantity':5},
                5: {'type':'log', 'quantity':1},
                6: {'type':'log', 'quantity':2},
                32: {'type':'iron_ore', 'quantity':4}
            )

        Args:
            inventory (Dict[int, Dict[str, Union[str,int]]]): The inventory slot description.
        """
        self.inventory = inventory