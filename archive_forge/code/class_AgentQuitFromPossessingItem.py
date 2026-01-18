from minerl.herobraine.hero.handler import Handler
import jinja2
from typing import List, Dict, Union
class AgentQuitFromPossessingItem(Handler):

    def to_string(self) -> str:
        return 'agent_quit_from_possessing_item'

    def xml_template(self) -> str:
        return str('<AgentQuitFromPossessingItem>\n                   {% for item in items %}\n                   <Item type="{{ item.type }}" amount="{{ item.amount }}"/>\n                   {% endfor %}\n               </AgentQuitFromPossessingItem>')

    def __init__(self, items: List[Dict[str, Union[str, int]]]):
        """Creates a reward which will cause the player to quit when they obtain something.
        
        aqfpi = AgentQuitFromPossessingItem([
            dict(type="log", amount=64)
        ])
        """
        assert isinstance(items, list)
        self.items = items
        for item in self.items:
            assert 'type' in item, '{} does contain `type`'.format(item)
            assert 'amount' in item, '{} does not contain `amount`'.format(item)