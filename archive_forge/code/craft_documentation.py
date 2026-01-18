from typing import Optional
from minerl.herobraine.hero.handlers.agent.action import Action, ItemListAction
import jinja2
import minerl.herobraine.hero.spaces as spaces

        Initializes the space of the handler to be one for each item in the list plus one for the
        default no-craft action (command 0)

        Items are minecraft resource ID's
        