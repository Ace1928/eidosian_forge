from minerl.herobraine.hero.handler import Handler
from typing import Dict, List, Union
import random
import jinja2
class StartingFoodAgentStart(Handler):

    def to_string(self) -> str:
        return 'starting_food_agent_start'

    def xml_template(self) -> str:
        if self.food_saturation is None:
            return str('<StartingFood food="{{ food }}"/>')
        else:
            return str('<StartingHealth food="{{ food }}" foodSaturation="{{ food_saturation }}"/>')

    def __init__(self, food: int=20, food_saturation: float=None):
        """Sets the starting food of the agent.

        For example:

            starting_health = StartingFoodAgentStart(2.5)

        Args:
            food: The amount of food the agent starts out with
            food_saturation: The food saturation the agent starts out with (if not specified, set to max)
        """
        self.food = food
        self.food_saturation = food_saturation